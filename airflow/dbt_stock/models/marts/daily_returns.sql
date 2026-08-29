-- Per-ticker daily return plus a 20-day rolling mean price and volatility.
-- Layered into CTEs because volatility_20d needs daily_return to already
-- exist as a column before it can be windowed again.
with staged as (
    select * from {{ ref('stg_prices') }}
),

with_return as (
    select
        *,
        close / lag(close) over (partition by ticker order by trade_date) - 1 as daily_return
    from staged
),

final as (
    select
        *,
        avg(close) over (
            partition by ticker order by trade_date
            rows between 19 preceding and current row
        ) as sma_20,
        stddev(daily_return) over (
            partition by ticker order by trade_date
            rows between 19 preceding and current row
        ) as volatility_20d
    from with_return
)

select * from final
