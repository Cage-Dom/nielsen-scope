CREATE TABLE IF NOT EXISTS categories (
    category varchar(1) PRIMARY KEY,
    category_name varchar(30)
);

CREATE TABLE IF NOT EXISTS books (
    isbn varchar(13) PRIMARY KEY,
    title varchar(200),
    author varchar(150),
    imprint varchar(100),
    publisher_group varchar(50),
    rrp DECIMAL(4,2),
    binding varchar(9),
    publication_date date,
    product_class varchar(80),
    country_of_publication varchar(25),
    category varchar(1) REFERENCES categories(category),
    data_version varchar(40) NOT NULL DEFAULT 'nielsen_raw',
    ingested_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sales (
    isbn varchar(13) NOT NULL REFERENCES books(isbn),
    reporting_interval varchar(6),
    end_date date NOT NULL,
    volume int,
    sale_value decimal(12,2),
    asp decimal(5,2),
    data_version varchar(40) NOT NULL DEFAULT 'nielsen_raw',
    ingested_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY(isbn, end_date, data_version)
);

CREATE INDEX IF NOT EXISTS idx_sales_isbn_week
ON sales(isbn, end_date);

CREATE TABLE IF NOT EXISTS forecasts (
    isbn         varchar(13)  NOT NULL,
    model_type   varchar(20)  NOT NULL
                 CHECK (model_type IN ('naive','xgboost','catboost','autoarima','crostonsba')),
    run_id       varchar(32)  NOT NULL,
    kind         varchar(8)   NOT NULL CHECK (kind IN ('backtest','future')),
    fold         smallint,
    end_date     date         NOT NULL,
    y_true       int,
    y_pred       double precision NOT NULL,
    lo_95        double precision,
    hi_95        double precision,
    data_version varchar(40)  NOT NULL DEFAULT 'nielsen_raw',
    ingested_at  timestamptz  NOT NULL DEFAULT now(),
    PRIMARY KEY (isbn, model_type, run_id, kind, end_date),
    CHECK ((kind = 'backtest') = (fold IS NOT NULL)),
    CHECK ((kind = 'backtest') = (y_true IS NOT NULL))
);

CREATE TABLE IF NOT EXISTS model_metrics (
    isbn         varchar(13)  NOT NULL,
    model_type   varchar(20)  NOT NULL
                 CHECK (model_type IN ('naive','xgboost','catboost','autoarima','crostonsba')),
    run_id       varchar(32)  NOT NULL,
    metric       varchar(24)  NOT NULL
                 CHECK (metric IN ('mae','mae_std','rmse','rmse_std','mape','mape_std',
                                   'rel_mae','mape_excluded_rows','one_step_mae')),
    fold         smallint,
    value        double precision,
    fell_back    boolean,
    data_version varchar(40)  NOT NULL DEFAULT 'nielsen_raw',
    ingested_at  timestamptz  NOT NULL DEFAULT now(),
    UNIQUE NULLS NOT DISTINCT (isbn, model_type, run_id, metric, fold)
);