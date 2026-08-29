-- Cleans and types the raw landed data before any business logic touches it.
select
    ticker,
    cast(date as date) as trade_date,
    open,
    high,
    low,
    close,
    volume
from {{ source('raw', 'raw_prices') }}
where close is not null
