-- Stock Market Platform – Database Schema
-- This runs automatically when the postgres container first starts.

CREATE TABLE IF NOT EXISTS live_stock_prices (
    id            SERIAL PRIMARY KEY,
    symbol        VARCHAR(10)    NOT NULL,
    timestamp     TIMESTAMPTZ    NOT NULL,
    price         NUMERIC(12,4)  NOT NULL,
    volume        BIGINT,
    open          NUMERIC(12,4),
    high          NUMERIC(12,4),
    low           NUMERIC(12,4),
    close         NUMERIC(12,4),
    created_at    TIMESTAMPTZ    DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_live_symbol    ON live_stock_prices (symbol);
CREATE INDEX IF NOT EXISTS idx_live_timestamp ON live_stock_prices (timestamp DESC);

CREATE TABLE IF NOT EXISTS analytics_daily_avg (
    id              SERIAL PRIMARY KEY,
    symbol          VARCHAR(10)   NOT NULL,
    date            DATE          NOT NULL,
    daily_avg_close NUMERIC(12,4),
    created_at      TIMESTAMPTZ   DEFAULT NOW(),
    UNIQUE (symbol, date)
);

CREATE TABLE IF NOT EXISTS analytics_monthly_trends (
    id               SERIAL PRIMARY KEY,
    symbol           VARCHAR(10)   NOT NULL,
    year             INT           NOT NULL,
    month            INT           NOT NULL,
    monthly_avg_close NUMERIC(12,4),
    created_at       TIMESTAMPTZ   DEFAULT NOW(),
    UNIQUE (symbol, year, month)
);

CREATE TABLE IF NOT EXISTS analytics_top_gainers (
    id         SERIAL PRIMARY KEY,
    symbol     VARCHAR(10)   NOT NULL,
    gain       NUMERIC(12,4),
    calculated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ml_predictions (
    id            SERIAL PRIMARY KEY,
    symbol        VARCHAR(10)   NOT NULL,
    predicted_at  TIMESTAMPTZ   DEFAULT NOW(),
    open_price    NUMERIC(12,4),
    high          NUMERIC(12,4),
    low           NUMERIC(12,4),
    volume        BIGINT,
    predicted_close NUMERIC(12,4),
    actual_close  NUMERIC(12,4),
    mae           NUMERIC(12,6)
);

CREATE INDEX IF NOT EXISTS idx_pred_symbol ON ml_predictions (symbol);

CREATE TABLE IF NOT EXISTS spike_alerts (
    id               SERIAL PRIMARY KEY,
    symbol           VARCHAR(10)   NOT NULL,
    alert_time       TIMESTAMPTZ   DEFAULT NOW(),
    price            NUMERIC(12,4),
    open_price       NUMERIC(12,4),
    price_change_pct NUMERIC(8,4),
    alert_type       VARCHAR(20)   DEFAULT 'spike'
);
