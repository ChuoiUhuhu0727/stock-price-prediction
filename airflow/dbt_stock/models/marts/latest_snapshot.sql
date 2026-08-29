-- One row per ticker: the most recent trading day's metrics. This is the
-- table a dashboard or the FastAPI service would read for "current" stats.
with ranked as (
    select
        *,
        row_number() over (partition by ticker order by trade_date desc) as rn
    from {{ ref('daily_returns') }}
)

select
    ticker,
    trade_date,
    close,
    daily_return,
    sma_20,
    volatility_20d
from ranked
where rn = 1
order by ticker
