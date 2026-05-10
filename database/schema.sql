CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY,
    created_at TIMESTAMP NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    setup_type TEXT,
    confidence REAL,
    probability REAL,
    entry REAL,
    sl REAL,
    tp REAL,
    volume REAL,
    spread_points REAL,
    slippage_points REAL,
    setup_score REAL,
    expected_r REAL,
    regime TEXT,
    execution_risk REAL,
    quality_tier TEXT,
    profit REAL,
    exit_r REAL,
    exit_price REAL,
    exit_reason TEXT,
    closed_at TIMESTAMP,
    status TEXT,
    execution_details TEXT,
    model_decision TEXT
);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    id INTEGER PRIMARY KEY,
    created_at TIMESTAMP NOT NULL,
    equity REAL NOT NULL,
    balance REAL,
    drawdown REAL
);

CREATE TABLE IF NOT EXISTS risk_events (
    id INTEGER PRIMARY KEY,
    created_at TIMESTAMP NOT NULL,
    event_type TEXT NOT NULL,
    message TEXT NOT NULL
);
