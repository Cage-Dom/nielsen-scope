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