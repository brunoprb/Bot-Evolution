import os
import json
import time
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any
from datetime import datetime
import math

import requests
import ccxt
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

# ==========================================
# 1. LOGGING (Registo de Atividades)
# ==========================================
logger = logging.getLogger("gateio_bot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)

# ==========================================
# 2. CONFIGURAÇÃO (Parâmetros do Bot)
# ==========================================
@dataclass(frozen=True)
class Config:
    gateio_api_key: str
    gateio_api_secret: str
    tg_bot_token: str
    tg_chat_id: str
    leverage: int = 3
    timeframe: str = "4h"
    trade_usdt: float = 5.0              # Valor fixo por operação (USD)
    max_open_positions: int = 10
    rsi_period: int = 14
    rsi_oversold: float = 40.0           # Gatilho de compra (abaixo disto é barato)
    rsi_overbought: float = 60.0         # Gatilho de venda (acima disto está caro)
    ema_period: int = 200                # Período para definir Alta ou Baixa do mercado
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    trailing_stop_pct: float = 0.04      # Margem do trailing
    trailing_activate_pct: float = 0.02  # Quando ativar o trailing de proteção
    top_symbols_limit: int = 200
    min_balance_usdt: float = 10.0
    max_symbol_price_usdt: float = 400.0
    cooldown_hours: float = 4.0          # Tempo de espera após fechar uma moeda
    data_dir: str = os.getenv("DATA_DIR", "./data")
    atr_period: int = 14                 # Para medir volatilidade
    daily_tp_pct: float = 0.015          # Meta de 1.5% de lucro diário

def load_config() -> Config:
    # Carrega as chaves do ambiente. Agora configurado para GATEIO.
    api_key = os.getenv("GATEIO_API_KEY", "").strip()
    api_secret = os.getenv("GATEIO_API_SECRET", "").strip()
    tg_token = os.getenv("TG_BOT_TOKEN", "").strip()
    tg_chat = os.getenv("TG_CHAT_ID", "").strip()

    if not all([api_key, api_secret, tg_token, tg_chat]):
        raise ValueError("❌ Faltam variáveis de ambiente! (GATEIO_API_KEY, GATEIO_API_SECRET, TG_BOT_TOKEN, TG_CHAT_ID)")

    return Config(
        gateio_api_key=api_key,
        gateio_api_secret=api_secret,
        tg_bot_token=tg_token,
        tg_chat_id=tg_chat,
        leverage=int(os.getenv("LEVERAGE", "3")),
        trade_usdt=float(os.getenv("TRADE_USDT", "5.0")),
        daily_tp_pct=float(os.getenv("DAILY_TP_PCT", "0.015"))
    )

@dataclass
class Position:
    symbol: str
    side: str
    entry: float
    stop: float
    take_profit: float
    amount: float
    opened_at: float
    usdt_invested: float
    peak: float = 0.0
    bottom: float = 0.0
    trailing_armed: bool = False
    source: str = "bot"

# ==========================================
# 3. CLASSE PRINCIPAL DO BOT
# ==========================================
class GateIoBot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        if not os.path.exists(self.cfg.data_dir):
            os.makedirs(self.cfg.data_dir, exist_ok=True)

        # Ficheiros de memória
        self.pos_path = os.path.join(self.cfg.data_dir, "positions_gateio.json")
        self.cooldown_path = os.path.join(self.cfg.data_dir, "cooldowns.json")
        self.daily_state_path = os.path.join(self.cfg.data_dir, "daily_state.json")

        self.exchange = self._init_exchange()
        self.daily_state = self.load_daily_state()
        self.positions: Dict[str, Position] = self.load_pos()
        self.cooldowns = self.load_cooldowns()
        self.offset = 0
        self.last_scan_time = 0

        try:
            self.sync_positions_from_exchange()
        except Exception as e:
            logger.warning(f"⚠️ Falha ao sincronizar posições da exchange: {e}")

    def _init_exchange(self) -> ccxt.gate:
        """Inicializa a conexão com a corretora Gate.io usando CCXT."""
        try:
            exchange = ccxt.gate({
                "apiKey": self.cfg.gateio_api_key,
                "secret": self.cfg.gateio_api_secret,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "swap",  # Importante para operar futuros (USDT-M)
                },
                "timeout": 30000
            })
            exchange.load_markets()
            logger.info("✅ Conectado à Gate.io!")
            return exchange
        except Exception as e:
            logger.error(f"❌ Erro ao conectar à Gate.io: {e}")
            raise

    # --- MEMÓRIA E ESTADO (JSON) ---
    def get_total_equity(self) -> float:
        try:
            bal = self.exchange.fetch_balance()
            return float(bal.get('total', {}).get('USDT', 0.0))
        except Exception as e:
            logger.warning(f"⚠️ Erro ao obter equity total: {e}")
            return 0.0

    def load_daily_state(self) -> dict:
        hoje = datetime.now().strftime("%Y-%m-%d")
        estado_padrao = {"data": hoje, "banca_inicial": 0.0, "meta_atingida": False}
        if os.path.exists(self.daily_state_path):
            try:
                with open(self.daily_state_path, "r") as f:
                    estado = json.load(f)
                    if estado.get("data") == hoje:
                        return estado
            except Exception:
                pass
        
        banca_atual = self.get_total_equity()
        estado_padrao["banca_inicial"] = banca_atual
        self.save_daily_state(estado_padrao)
        return estado_padrao

    def save_daily_state(self, estado=None):
        if estado is None:
            estado = self.daily_state
        try:
            with open(self.daily_state_path, "w") as f:
                json.dump(estado, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Erro ao salvar daily state: {e}")

    def check_daily_reset(self):
        hoje = datetime.now().strftime("%Y-%m-%d")
        if self.daily_state.get("data") != hoje:
            logger.info("🕛 Meia-noite! Resetando a banca inicial do dia.")
            banca_atual = self.get_total_equity()
            self.daily_state = {
                "data": hoje,
                "banca_inicial": banca_atual,
                "meta_atingida": False
            }
            self.save_daily_state()
            self.send_tg(f"🕛 *Novo Dia de Operações!*\nBanca inicial gravada: ${banca_atual:.2f}")

    def check_daily_take_profit(self):
        if self.daily_state.get("meta_atingida"): return
        banca_inicial = self.daily_state.get("banca_inicial", 0.0)
        if banca_inicial <= 0: return

        banca_atual = self.get_total_equity()
        if banca_atual <= 0: return

        lucro_percentual = (banca_atual - banca_inicial) / banca_inicial
        if lucro_percentual >= self.cfg.daily_tp_pct:
            msg = (f"🎯 *META DIÁRIA ATINGIDA!*\n"
                   f"Lucro alcançado: {lucro_percentual*100:.2f}%\n"
                   f"Fechando tudo e pausando até amanhã.")
            logger.info(msg)
            self.send_tg(msg)
            self.close_all()
            self.daily_state["meta_atingida"] = True
            self.save_daily_state()

    def load_pos(self) -> Dict[str, Position]:
        if not os.path.exists(self.pos_path): return {}
        try:
            with open(self.pos_path, "r") as f:
                data = json.load(f)
                positions = {}
                for s, p in data.items():
                    if "side" not in p: p["side"] = "long"
                    if "bottom" not in p: p["bottom"] = p.get("entry", 0.0)
                    if "take_profit" not in p:
                        entry = p.get("entry", 0.0)
                        p["take_profit"] = entry * (1.0 + self.cfg.take_profit_pct) if p["side"] == "long" else entry * (1.0 - self.cfg.take_profit_pct)
                    positions[s] = Position(**p)
                return positions
        except Exception:
            return {}

    def save_pos(self):
        try:
            with open(self.pos_path, "w") as f:
                json.dump({s: asdict(p) for s, p in self.positions.items()}, f, indent=2)
        except Exception as e:
            logger.error(f"Erro ao salvar posições: {e}")

    def load_cooldowns(self) -> Dict[str, float]:
        if not os.path.exists(self.cooldown_path): return {}
        try:
            with open(self.cooldown_path, "r") as f: return json.load(f)
        except Exception: return {}

    def save_cooldowns(self):
        try:
            with open(self.cooldown_path, "w") as f: json.dump(self.cooldowns, f, indent=2)
        except Exception: pass

    def is_in_cooldown(self, symbol: str) -> bool:
        if symbol not in self.cooldowns: return False
        if time.time() < self.cooldowns[symbol] + (self.cfg.cooldown_hours * 3600): return True
        del self.cooldowns[symbol]
        self.save_cooldowns()
        return False

    def add_cooldown(self, symbol: str):
        self.cooldowns[symbol] = time.time()
        self.save_cooldowns()

    # --- TELEGRAM ---
    def send_tg(self, msg: str):
        msg_escaped = msg.replace("_", "\\_").replace("[", "\\[").replace("]", "\\]")
        for _ in range(3):
            try:
                url = f"https://api.telegram.org/bot{self.cfg.tg_bot_token}/sendMessage"
                r = requests.post(url, json={"chat_id": self.cfg.tg_chat_id, "text": msg_escaped, "parse_mode": "Markdown"}, timeout=10)
                r.raise_for_status()
                return
            except Exception: time.sleep(2)

    def handle_commands(self):
        try:
            url = f"https://api.telegram.org/bot{self.cfg.tg_bot_token}/getUpdates"
            r = requests.get(url, params={"offset": self.offset, "timeout": 1}, timeout=5)
            for upd in r.json().get("result", []):
                self.offset = upd["update_id"] + 1
                text = upd.get("message", {}).get("text", "").lower().strip()
                if text == "/status":
                    self.send_tg(f"💰 Saldo: ${self.get_total_equity():.2f}\nPosições: {len(self.positions)}")
                elif text in ["/fechar_tudo", "/panic_close"]:
                    self.close_all()
        except Exception: pass

    # --- OPERAÇÕES NA CORRETORA ---
    def close_all(self):
        self.send_tg("⚠️ *FECHANDO TUDAS AS POSIÇÕES*")
        for s, p in list(self.positions.items()):
            try:
                params = {"reduceOnly": True}
                if p.side == "long":
                    self.exchange.create_market_sell_order(s, p.amount, params)
                else:
                    self.exchange.create_market_buy_order(s, p.amount, params)
                self.add_cooldown(s)
            except Exception as e:
                logger.error(f"❌ Erro ao fechar {s}: {e}")
        self.positions.clear()
        self.save_pos()

    def _close_position(self, symbol: str, pos: Position, exit_price: float, reason: str):
        try:
            params = {"reduceOnly": True}
            if pos.side == "long":
                self.exchange.create_market_sell_order(symbol, pos.amount, params)
                pnl = (exit_price - pos.entry) * pos.amount
            else:
                self.exchange.create_market_buy_order(symbol, pos.amount, params)
                pnl = (pos.entry - exit_price) * pos.amount

            pnl_pct = (pnl / pos.usdt_invested) * 100 if pos.usdt_invested else 0.0
            msg = f"🎯 *{symbol} {pos.side.upper()} FECHADO*\nMotivo: {reason}\nPnL: ${pnl:+.2f} ({pnl_pct:+.2f}%)"
            self.send_tg(msg)
            logger.info(msg)

            self.add_cooldown(symbol)
            if symbol in self.positions: del self.positions[symbol]
            self.save_pos()
        except Exception as e:
            logger.error(f"❌ Erro ao fechar {symbol}: {e}")

    # --- LÓGICA DE MONITORIZAÇÃO E TENDÊNCIA ---
    def monitor(self):
        """Avalia constantemente as posições abertas para ativar Take Profit ou Stop Loss."""
        for s, p in list(self.positions.items()):
            try:
                ticker = self.exchange.fetch_ticker(s)
                last = float(ticker['last'])

                if p.side == "long":
                    if last > p.peak: p.peak = last
                    if not p.trailing_armed and last >= p.entry * (1.0 + self.cfg.trailing_activate_pct):
                        p.trailing_armed = True
                    stop_price = p.peak * (1.0 - self.cfg.trailing_stop_pct) if p.trailing_armed else p.stop
                    if last >= p.take_profit: self._close_position(s, p, last, "TAKE PROFIT")
                    elif last <= stop_price: self._close_position(s, p, last, "STOP LOSS/TRAILING")
                else:
                    if p.bottom == 0.0 or last < p.bottom: p.bottom = last
                    if not p.trailing_armed and last <= p.entry * (1.0 - self.cfg.trailing_activate_pct):
                        p.trailing_armed = True
                    stop_price = p.bottom * (1.0 + self.cfg.trailing_stop_pct) if p.trailing_armed else p.stop
                    if last <= p.take_profit: self._close_position(s, p, last, "TAKE PROFIT")
                    elif last >= stop_price: self._close_position(s, p, last, "STOP LOSS/TRAILING")
            except Exception as e:
                logger.warning(f"⚠️ Erro monitorizando {s}: {e}")

    def scan(self):
        """Busca novas oportunidades de entrada filtrando Mercado em Alta vs Baixa."""
        if self.daily_state.get("meta_atingida") or time.time() - self.last_scan_time < 60: return
        self.last_scan_time = time.time()
        if len(self.positions) >= self.cfg.max_open_positions: return

        try:
            markets = self.exchange.load_markets()
            # Gate.io usa a formatação MOEDA/USDT:USDT para os futuros lineares
            symbols = [s for s, m in markets.items() if m.get('active') and m.get('linear')]

            for s in symbols[:self.cfg.top_symbols_limit]:
                if len(self.positions) >= self.cfg.max_open_positions: break
                if s in self.positions or self.is_in_cooldown(s): continue

                try:
                    ticker = self.exchange.fetch_ticker(s)
                    price = float(ticker.get('last', 0.0))
                    if price <= 0 or price > self.cfg.max_symbol_price_usdt: continue

                    # Baixamos dados de preço para análise técnica
                    ohlcv = self.exchange.fetch_ohlcv(s, self.cfg.timeframe, limit=self.cfg.ema_period + 10)
                    if not ohlcv or len(ohlcv) < self.cfg.ema_period: continue

                    df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
                    df['c'] = pd.to_numeric(df['c'])

                    # Indicadores Técnicos
                    rsi = RSIIndicator(df['c'], self.cfg.rsi_period).rsi().iloc[-1]
                    ema_trend = EMAIndicator(df['c'], self.cfg.ema_period).ema_indicator().iloc[-1]
                    atr = AverageTrueRange(df['h'], df['l'], df['c'], self.cfg.atr_period).average_true_range().iloc[-1]

                    # MUDANÇA PRINCIPAL: LÓGICA DE ALTA (BULL) vs BAIXA (BEAR)
                    is_bull_market = price > ema_trend
                    is_bear_market = price < ema_trend

                    # Mercado em ALTA: Procuramos mergulhos rápidos (RSI sobrevendido) para comprar (Long)
                    if is_bull_market and rsi < self.cfg.rsi_oversold:
                        logger.debug(f"Sinal LONG de TENDÊNCIA ALTA em {s}")
                        self._open_position(s, price, "long", atr)

                    # Mercado em BAIXA: Procuramos saltos rápidos (RSI sobrecomprado) para vender a descoberto (Short)
                    elif is_bear_market and rsi > self.cfg.rsi_overbought:
                        logger.debug(f"Sinal SHORT de TENDÊNCIA BAIXA em {s}")
                        self._open_position(s, price, "short", atr)

                    time.sleep(0.1) # Evita limites da API
                except Exception as e:
                    continue
        except Exception as e:
            logger.error(f"❌ Erro no scan: {e}")

    def _open_position(self, symbol: str, price: float, side: str, atr: float):
        try:
            trade_usdt = self.cfg.trade_usdt
            notional = trade_usdt * self.cfg.leverage
            qty = notional / price
            try:
                qty_precise = float(self.exchange.amount_to_precision(symbol, qty))
            except Exception:
                qty_precise = float(math.floor(qty * 1e8) / 1e8)

            # Gate.io setup de Alavancagem
            try:
                self.exchange.set_leverage(self.cfg.leverage, symbol)
            except Exception: pass

            atr_pct = atr / price if price > 0 else self.cfg.stop_loss_pct
            stop_pct = max(self.cfg.stop_loss_pct, min(atr_pct * 1.5, 0.20)) # Stop adaptável

            if side == "long":
                stop = price * (1.0 - stop_pct)
                tp = price * (1.0 + self.cfg.take_profit_pct)
                order = self.exchange.create_market_buy_order(symbol, qty_precise)
            else:
                stop = price * (1.0 + stop_pct)
                tp = price * (1.0 - self.cfg.take_profit_pct)
                order = self.exchange.create_market_sell_order(symbol, qty_precise)

            entry = float(order.get('average') or order.get('price') or price)
            
            pos = Position(symbol=symbol, side=side, entry=entry, stop=stop, take_profit=tp,
                           amount=qty_precise, opened_at=time.time(), usdt_invested=trade_usdt,
                           peak=entry if side=="long" else 0.0, bottom=entry if side=="short" else 0.0)

            self.positions[symbol] = pos
            self.save_pos()
            self.send_tg(f"🚀 *{side.upper()} ABERTO*\nMoeda: {symbol}\nPreço: ${entry:.4f}")
        except Exception as e:
            logger.error(f"❌ Erro abrindo posição {side} em {symbol}: {e}")

    # --- O CÓDIGO QUE FICOU CORTADO ANTES ---
    def fetch_open_positions_from_exchange(self) -> list:
        positions = []
        try:
            if hasattr(self.exchange, "fetch_positions"):
                pos_list = self.exchange.fetch_positions()
                for p in pos_list:
                    size = float(p.get('contracts') or p.get('size') or p.get('positionAmt') or 0)
                    if size != 0:
                        sym = p.get('symbol') or p.get('info', {}).get('contract')
                        side = p.get('side')
                        if side is None:
                            side = "long" if size > 0 else "short"
                        positions.append({"symbol": sym, "size": abs(size), "side": side.lower(), "raw": p})
        except Exception as e:
            logger.warning(f"⚠️ Erro ao buscar posições externas: {e}")
        return positions

    def sync_positions_from_exchange(self):
        remote = self.fetch_open_positions_from_exchange()
        if not remote: return

        for rp in remote:
            sym = rp.get("symbol")
            if not sym or sym in self.positions: continue
            
            size = float(rp.get("size", 0))
            if size <= 0: continue
            
            side = rp.get("side", "long")
            raw = rp.get("raw", {})
            entry = 0.0
            
            # Procura o preço de entrada numa lista de chaves possíveis
            for k in ('entryPrice', 'avgEntryPrice', 'avgPrice', 'price', 'markPrice'):
                if raw.get(k):
                    entry = float(raw[k])
                    break
                elif raw.get('info', {}).get(k):
                    entry = float(raw['info'][k])
                    break
            
            if entry > 0:
                pos = Position(
                    symbol=sym, side=side, entry=entry,
                    stop=entry * 0.9 if side=="long" else entry * 1.1,
                    take_profit=entry * 1.1 if side=="long" else entry * 0.9,
                    amount=size, opened_at=time.time(), usdt_invested=size * entry / self.cfg.leverage,
                    peak=entry, bottom=entry, source="sync"
                )
                self.positions[sym] = pos
        self.save_pos()

    def run(self):
        """Ciclo principal de vida do programa (Main Loop)."""
        logger.info("🟢 Inicializando Bot Gate.io...")
        self.send_tg("✅ *Bot Iniciado e conectado à Gate.io!*")
        while True:
            try:
                self.handle_commands()
                self.check_daily_reset()
                self.check_daily_take_profit()
                self.monitor()
                self.scan()
                time.sleep(15)  # Pequena pausa para não sobrecarregar CPU e API
            except KeyboardInterrupt:
                logger.info("🛑 Bot parado pelo utilizador.")
                break
            except Exception as e:
                logger.error(f"❌ Erro crítico no loop principal: {e}")
                time.sleep(30)

# ==========================================
# 4. EXECUÇÃO
# ==========================================
if __name__ == "__main__":
    try:
        config = load_config()
        bot = GateIoBot(config)
        bot.run()
    except Exception as e:
        logger.error(f"Falha ao iniciar: {e}")
