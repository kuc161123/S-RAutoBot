"""
Backtest persistence utilities (PostgreSQL via SQLAlchemy)

Defines minimal tables to store pretraining/backtest runs and per-trade results.
Reuses DATABASE_URL from environment (falls back to SQLite like candle storage).
"""
from __future__ import annotations

import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    create_engine, Column, Integer, BigInteger, Float, String, DateTime, Text, JSON, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.pool import NullPool

logger = logging.getLogger(__name__)

Base = declarative_base()


class BacktestRun(Base):
    __tablename__ = 'backtest_runs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    finished_at = Column(DateTime)
    universe = Column(Text)  # JSON list as text for compatibility
    config = Column(Text)    # JSON dict as text for compatibility
    notes = Column(Text)
    signals = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    wr_percent = Column(Float, default=0.0)
    avg_r = Column(Float, default=0.0)
    ev_r = Column(Float, default=0.0)
    variant_label = Column(String(120))

    trades = relationship('BacktestTrade', back_populates='run', cascade='all, delete-orphan')


class BacktestTrade(Base):
    __tablename__ = 'backtest_trades'
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey('backtest_runs.id'), nullable=False)
    symbol = Column(String(24), nullable=False)
    side = Column(String(8), nullable=False)
    entry = Column(Float, nullable=False)
    sl = Column(Float, nullable=False)
    tp = Column(Float, nullable=False)
    entry_time = Column(DateTime, nullable=False)
    outcome = Column(String(12), nullable=False)  # win|loss
    realized_r = Column(Float, nullable=False)
    meta = Column(Text)  # JSON text (divergence, breakout_level, etc.)

    run = relationship('BacktestRun', back_populates='trades')


class BacktestPersistence:
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.getenv('DATABASE_URL')
        self.is_postgres = False
        if not self.database_url:
            # SQLite fallback
            self.database_url = f"sqlite:///{os.path.join(os.path.dirname(os.path.abspath(__file__)), 'candles.db')}"
        else:
            if self.database_url.startswith('postgresql://'):
                self.database_url = self.database_url.replace('postgresql://', 'postgresql+psycopg2://')
            self.is_postgres = True
        self.engine = create_engine(
            self.database_url,
            poolclass=NullPool,
            connect_args={} if self.is_postgres else {"check_same_thread": False}
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def start_run(self, universe: List[str], config: Dict[str, Any], variant_label: str) -> int:
        s = self.Session()
        try:
            run = BacktestRun(
                universe=json.dumps(universe),
                config=json.dumps(config),
                variant_label=variant_label
            )
            s.add(run); s.commit()
            return int(run.id)
        finally:
            s.close()

    def finish_run(self, run_id: int, signals: int, wins: int, losses: int, avg_r: float, ev_r: float, notes: str = "") -> None:
        s = self.Session()
        try:
            run = s.get(BacktestRun, run_id)
            if not run:
                return
            run.finished_at = datetime.utcnow()
            run.signals = int(signals)
            run.wins = int(wins)
            run.losses = int(losses)
            run.wr_percent = (wins / max(1, signals)) * 100.0
            run.avg_r = float(avg_r)
            run.ev_r = float(ev_r)
            if notes:
                run.notes = (run.notes + "\n" + notes) if run.notes else notes
            s.commit()
        finally:
            s.close()

    def add_trades(self, run_id: int, trades: List[Dict[str, Any]]):
        if not trades:
            return
        s = self.Session()
        try:
            for t in trades:
                bt = BacktestTrade(
                    run_id=run_id,
                    symbol=str(t['symbol']),
                    side=str(t['side']),
                    entry=float(t['entry']),
                    sl=float(t['sl']),
                    tp=float(t['tp']),
                    entry_time=t['entry_time'] if isinstance(t['entry_time'], datetime) else datetime.fromisoformat(str(t['entry_time'])),
                    outcome=str(t['outcome']),
                    realized_r=float(t['realized_r']),
                    meta=json.dumps(t.get('meta', {}))
                )
                s.add(bt)
            s.commit()
        finally:
            s.close()

