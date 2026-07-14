-- Per-(isbn, week) model-ready feature bank for Nielsen weekly sales forecast.
-- Target = volume; predictors = volume lags/trailing rolling means, lagged
-- price signals, calendar parts, and static book metadata. All autoregressive
-- work is done in SQL via window functions over the full nielsen_raw fact.
-- Kept book/model agnostic to develop one fixed, generous feature set for
-- every book.
SELECT
    s.isbn,
    s.end_date,
    s.volume,                                            -- target: weekly units sold
    -- Autoregressive lags = volume N weeks ago. LAG(col, N) OVER w reads the
    -- value N rows back within the window w. 1-4 recent momentum, 8/13/26
    -- sub-annual, 52 = same week last year (annual seasonality).
    LAG(s.volume, 1)  OVER w AS volume_lag_1,
    LAG(s.volume, 2)  OVER w AS volume_lag_2,
    LAG(s.volume, 3)  OVER w AS volume_lag_3,
    LAG(s.volume, 4)  OVER w AS volume_lag_4,
    LAG(s.volume, 8)  OVER w AS volume_lag_8,
    LAG(s.volume, 13) OVER w AS volume_lag_13,
    LAG(s.volume, 26) OVER w AS volume_lag_26,
    LAG(s.volume, 52) OVER w AS volume_lag_52,
    -- trailing rolling means, EXCLUDING the current week (no target leakage)
    AVG(s.volume) OVER (w ROWS BETWEEN 4  PRECEDING AND 1 PRECEDING) AS volume_roll_4,
    AVG(s.volume) OVER (w ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING) AS volume_roll_12,
    -- price signals, LAGGED only: same-week asp/value are 0 on zero-filled
    -- no-sale weeks and move with the target, so they would leak it
    LAG(s.asp, 1)        OVER w AS asp_lag_1,
    LAG(s.sale_value, 1) OVER w AS sale_value_lag_1,
    -- EXTRACT pulls one calendar part out of a date; ::int casts it to a whole number
    EXTRACT(WEEK    FROM s.end_date)::int AS week_of_year,
    EXTRACT(MONTH   FROM s.end_date)::int AS month,
    EXTRACT(QUARTER FROM s.end_date)::int AS quarter,
    EXTRACT(YEAR    FROM s.end_date)::int AS year,
    -- static book metadata (constant per ISBN; raw — encoding is a modelling step)
    b.category,                                          -- b.<col> = a column from the 'books' dimension
    b.rrp,
    b.binding,
    b.publisher_group,
    b.product_class
FROM sales s
JOIN books b ON b.isbn = s.isbn
WHERE s.data_version = 'nielsen_raw'                     -- initial dataset
WINDOW w AS (PARTITION BY s.isbn ORDER BY s.end_date)    -- the window: per book, in week order
ORDER BY s.isbn, s.end_date;