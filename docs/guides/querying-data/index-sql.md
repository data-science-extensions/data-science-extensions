# Data Querying for Data Scientists


### A Comprehensive Guide of using Pandas, SQL, PySpark, and Polars for Data Manipulation Techniques, with Practical Examples and Visualisations

If you wanted to run the code yourself, you can download just that Jupyter notebook:

[![](./../../assets/images/dse.png){width=70px}<br>Download<br>- ALL](./index-r.ipynb){ :download .md-button .md-button-fifth }
[![](./../../assets/icons/pandas.svg){width=100%}<br>Download<br>- Pandas](./index-pandas-r.ipynb){ :download .md-button .md-button-fifth }
[![](./../../assets/icons/sql.svg){width=100%}<br>Download<br>- SQL](./index-sql-r.ipynb){ :download .md-button .md-button-fifth }
[![](./../../assets/icons/spark.svg){width=100%}<br>Download<br>- PySpark](./index-pyspark-r.ipynb){ :download .md-button .md-button-fifth }
[![](./../../assets/icons/polars.svg){width=100%}<br>Download<br>- Polars](./index-polars-r.ipynb){ :download .md-button .md-button-fifth }

Or you can follow along on this page...


## Introduction

Working as a Data Scientist or Data Engineer often involves querying data from various sources. There are many tools and libraries available to perform these tasks, each with its own strengths and weaknesses. Also, there are many different ways to achieve similar results, depending on the tool or library used. It's important to be familiar with these different methods to choose the best one for your specific use case.

This article provides a comprehensive guide on how to query data using different tools and libraries, including Pandas, SQL, PySpark, and Polars. Each section will cover the setup, data creation, and various querying techniques such as filtering, grouping, joining, window functions, ranking, and sorting. The output will be identical across all tools, but the transformations will be implemented using the specific syntax and features of each library. Therefore allowing you to compare the different approaches and understand the nuances of each method.


## Overview of the Different Libraries

Before we dive into the querying techniques, let's take a moment to understand the different libraries and tools we will be using in this article. Each library has its own strengths and weaknesses, and understanding these can help you choose the right tool for your specific use case.

Throughout this article, you can easily switch between the different libraries by selecting the appropriate tab. Each section will provide the same functionality, but implemented using the specific syntax and features of each library.

=== "SQL"

    [SQL][sql-wiki] (Structured Query Language) is a standard language for managing and manipulating relational databases. It is widely used for querying and modifying data in databases. SQL is a declarative language, meaning that you specify what you want to retrieve or manipulate without detailing how to do it. This makes SQL queries concise and expressive. SQL is particularly well-suited for working with large datasets and complex queries. It provides powerful features for filtering, grouping, joining, and aggregating data. SQL is the backbone of many database systems.

    SQL is actually a language (like Python is a language), not a library (like Pandas is a library), and it is used to interact with relational databases. The core of the SQL language is actually an [ISO standard][sql-iso], which means that the basic syntax and functionality are consistent across different database systems. However, each database system may have its own extensions or variations of SQL, which can lead to differences in syntax and features. Each database system can be considered as variations (or dialects) of SQL, with their own specific features and optimizations and syntax ehnancements.

    Some of the more popular SQL dialects include:

    <div class="mdx-three-columns" markdown>

    - [SQLite]
    - [PostgreSQL]
    - [MySQL]
    - [Spark SQL][spark-sql]
    - [SQL Server (t-SQL)][t-sql]
    - [Oracle SQL (pl-SQL)][pl-sql]

    </div>

## Setup

Before we start querying data, we need to set up our environment. This includes importing the necessary libraries, creating sample data, and defining constants that will be used throughout the article. The following sections will guide you through this setup process. The code for this article is also available on GitHub: [querying-data][querying-data].

=== "SQL"

    ```py {.sql linenums="1" title="Setup"}
    # StdLib Imports
    import sqlite3
    from typing import Any

    # Third Party Imports
    import numpy as np
    import pandas as pd
    from plotly import express as px, graph_objects as go, io as pio


    # Set default Plotly template
    pio.templates.default = "simple_white+gridon"

    # Set Pandas display options
    pd.set_option("display.max_columns", None)
    ```

Once the setup is complete, we can proceed to create our sample data. This data will be used for querying and will be consistent across all libraries. All tables will be created from scratch with randomly generated data to simulate a real-world scenario. This is to ensure that the examples are self-contained and can be run without any external dependencies, and also there is no issues about data privacy or security.

For the below data creation steps, we will be defining the tables using Python dictionaries. Each dictionary will represent a table, with keys as column names and values as lists of data. We will then convert these dictionaries into DataFrames or equivalent structures in each library.

First, we will create a sales fact table. This table will contain information about sales transactions, including the date, customer ID, product ID, category, sales amount, and quantity sold.

```py {.python linenums="1" title="Create Sales Fact Data"}
sales_data: dict[str, Any] = {
    "date": pd.date_range(start="2023-01-01", periods=n_records, freq="D"),
    "customer_id": np.random.randint(1, 100, n_records),
    "product_id": np.random.randint(1, 50, n_records),
    "category": np.random.choice(["Electronics", "Clothing", "Food", "Books", "Home"], n_records),
    "sales_amount": np.random.uniform(10, 1000, n_records).round(2),
    "quantity": np.random.randint(1, 10, n_records),
}
```

Next, we will create a product dimension table. This table will contain information about products, including the product ID, name, price, category, and supplier ID.

```py {.python linenums="1" title="Create Product Dimension Data"}
product_data: dict[str, Any] = {
    "product_id": np.arange(1, 51),
    "product_name": [f"Product {i}" for i in range(1, 51)],
    "price": np.random.uniform(10, 500, 50).round(2),
    "category": np.random.choice(["Electronics", "Clothing", "Food", "Books", "Home"], 50),
    "supplier_id": np.random.randint(1, 10, 50),
}
```

Finally, we will create a customer dimension table. This table will contain information about customers, including the customer ID, name, city, state, and segment.

```py {.python linenums="1" title="Create Customer Dimension Data"}
customer_data: dict[str, Any] = {
    "customer_id": np.arange(1, 101),
    "customer_name": [f"Customer {i}" for i in range(1, 101)],
    "city": np.random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"], 100),
    "state": np.random.choice(["NY", "CA", "IL", "TX", "AZ"], 100),
    "segment": np.random.choice(["Consumer", "Corporate", "Home Office"], 100),
}
```

Now that we have our sample data created, we can proceed to the querying section. Each of the following sections will demonstrate how to perform similar operations using the different libraries and methods, allowing you to compare and contrast their capabilities.


## Create the DataFrames

=== "SQL"

    To create the dataframes in SQL, we will use the data we generated earlier. Firstly, we need to create the SQLite database. This will be an in-memory database for demonstration purposes, but in a real-world scenario, you would typically connect to a persistent (on-disk) database. To do this, we will use the [`sqlite3`][sqlite3] library to create a connection to the database, which we define with the `:memory:` parameter on the [`.connect()`][sqlite3-connect] function. The result is to create a temporary database that exists only during the lifetime of the connection.

    Next, we will then parse the dictionaries into Pandas DataFrames, which will then be loaded into an SQLite database. This allows us to perform various data manipulation tasks using SQL queries.

    ```py {.sql linenums="1" title="Create DataFrames"}
    # Creates SQLite database and tables
    conn: sqlite3.Connection = sqlite3.connect(":memory:")
    pd.DataFrame(sales_data).to_sql("sales", conn, index=False, if_exists="replace")
    pd.DataFrame(product_data).to_sql("product", conn, index=False, if_exists="replace")
    pd.DataFrame(customer_data).to_sql("customer", conn, index=False, if_exists="replace")
    ```

    Once the data is created, we can check that it has been loaded correctly by displaying the first few rows of each DataFrame. To do this, we will use the [`pd.read_sql()`][pandas-read_sql] function to execute SQL queries and retrieve the data from the database. We will then parse the results to the [`print()`][python-print] function to display the DataFrame in a readable format.

    ```py {.sql linenums="1" title="Check Sales DataFrame"}
    print(f"Sales Table: {len(pd.read_sql('SELECT * FROM sales', conn))}")
    print(pd.read_sql("SELECT * FROM sales LIMIT 5", conn))
    print(pd.read_sql("SELECT * FROM sales LIMIT 5", conn).to_markdown())
    ```

    <div class="result" markdown>

    ```txt
    Sales Table: 100
    ```

    ```txt
                      date  customer_id  product_id     category  sales_amount  quantity
    0  2023-01-01 00:00:00           52          45         Food        490.76         7
    1  2023-01-02 00:00:00           93          41  Electronics        453.94         5
    2  2023-01-03 00:00:00           15          29         Home        994.51         5
    3  2023-01-04 00:00:00           72          15  Electronics        184.17         7
    4  2023-01-05 00:00:00           61          45         Food         27.89         9
    ```

    |      | date                | customer_id | product_id | category    | sales_amount | quantity |
    | ---: | :------------------ | ----------: | ---------: | :---------- | -----------: | -------: |
    |    0 | 2023-01-01 00:00:00 |          52 |         45 | Food        |       490.76 |        7 |
    |    1 | 2023-01-02 00:00:00 |          93 |         41 | Electronics |       453.94 |        5 |
    |    2 | 2023-01-03 00:00:00 |          15 |         29 | Home        |       994.51 |        5 |
    |    3 | 2023-01-04 00:00:00 |          72 |         15 | Electronics |       184.17 |        7 |
    |    4 | 2023-01-05 00:00:00 |          61 |         45 | Food        |        27.89 |        9 |

    </div>

    ```py {.sql linenums="1" title="Check Product DataFrame"}
    print(f"Product Table: {len(pd.read_sql('SELECT * FROM product', conn))}")
    print(pd.read_sql("SELECT * FROM product LIMIT 5", conn))
    print(pd.read_sql("SELECT * FROM product LIMIT 5", conn).to_markdown())
    ```

    <div class="result" markdown>

    ```txt
    Product Table: 50
    ```

    ```txt
       product_id product_name   price  category  supplier_id
    0           1    Product 1  257.57      Food            8
    1           2    Product 2  414.96  Clothing            5
    2           3    Product 3  166.82  Clothing            8
    3           4    Product 4  448.81      Food            4
    4           5    Product 5  200.71      Food            8
    ```

    |      | product_id | product_name |  price | category | supplier_id |
    | ---: | ---------: | :----------- | -----: | :------- | ----------: |
    |    0 |          1 | Product 1    | 257.57 | Food     |           8 |
    |    1 |          2 | Product 2    | 414.96 | Clothing |           5 |
    |    2 |          3 | Product 3    | 166.82 | Clothing |           8 |
    |    3 |          4 | Product 4    | 448.81 | Food     |           4 |
    |    4 |          5 | Product 5    | 200.71 | Food     |           8 |

    </div>

    ```py {.sql linenums="1" title="Check Customer DataFrame"}
    print(f"Customer Table: {len(pd.read_sql('SELECT * FROM customer', conn))}")
    print(pd.read_sql("SELECT * FROM customer LIMIT 5", conn))
    print(pd.read_sql("SELECT * FROM customer LIMIT 5", conn).to_markdown())
    ```

    <div class="result" markdown>

    ```txt
    Customer Table: 100
    ```

    ```txt
       customer_id customer_name         city state      segment
    0            1    Customer 1      Phoenix    NY    Corporate
    1            2    Customer 2      Phoenix    CA  Home Office
    2            3    Customer 3      Phoenix    NY  Home Office
    3            4    Customer 4  Los Angeles    NY     Consumer
    4            5    Customer 5  Los Angeles    IL  Home Office
    ```

    |      | customer_id | customer_name | city        | state | segment     |
    | ---: | ----------: | :------------ | :---------- | :---- | :---------- |
    |    0 |           1 | Customer 1    | Phoenix     | NY    | Corporate   |
    |    1 |           2 | Customer 2    | Phoenix     | CA    | Home Office |
    |    2 |           3 | Customer 3    | Phoenix     | NY    | Home Office |
    |    3 |           4 | Customer 4    | Los Angeles | NY    | Consumer    |
    |    4 |           5 | Customer 5    | Los Angeles | IL    | Home Office |

    </div>

## 1. Filtering and Selecting

This first section will demonstrate how to filter and select data from the DataFrames. This is a common operation in data analysis, allowing us to focus on specific subsets of the data.

=== "SQL"

    In SQL, we can use the `WHERE` clause to filter rows based on specific conditions. The syntax should be very familiar to anyone who has worked with SQL before. We can use the [`pd.read_sql()`][pandas-read_sql] function to execute SQL queries and retrieve the data from the database. The result is a Pandas DataFrame that contains only the rows that match the specified condition. In the below example, we filter for sales in the "Electronics" category.

    For more information about filtering in SQL, see the [SQL WHERE clause documentation][sqlite-where].

    ```py {.sql linenums="1" title="Filter sales for a specific category"}
    electronics_sales_txt: str = """
        SELECT *
        FROM sales
        WHERE category = 'Electronics'
    """
    electronics_sales_sql: pd.DataFrame = pd.read_sql(electronics_sales_txt, conn)
    print(f"Number of Electronics Sales: {len(electronics_sales_sql)}")
    print(pd.read_sql(electronics_sales_txt + "LIMIT 5", conn))
    print(pd.read_sql(electronics_sales_txt + "LIMIT 5", conn).to_markdown())
    ```

    <div class="result" markdown>

    ```txt
    Number of Electronics Sales: 28
    ```

    ```txt
                      date  customer_id  product_id     category  sales_amount  quantity
    0  2023-01-02 00:00:00           93          41  Electronics        453.94         5
    1  2023-01-04 00:00:00           72          15  Electronics        184.17         7
    2  2023-01-09 00:00:00           75           9  Electronics        746.73         2
    3  2023-01-11 00:00:00           88           1  Electronics        314.98         9
    4  2023-01-12 00:00:00           24          44  Electronics        547.11         8
    ```

    |      | date                | customer_id | product_id | category    | sales_amount | quantity |
    | ---: | :------------------ | ----------: | ---------: | :---------- | -----------: | -------: |
    |    0 | 2023-01-02 00:00:00 |          93 |         41 | Electronics |       453.94 |        5 |
    |    1 | 2023-01-04 00:00:00 |          72 |         15 | Electronics |       184.17 |        7 |
    |    2 | 2023-01-09 00:00:00 |          75 |          9 | Electronics |       746.73 |        2 |
    |    3 | 2023-01-11 00:00:00 |          88 |          1 | Electronics |       314.98 |        9 |
    |    4 | 2023-01-12 00:00:00 |          24 |         44 | Electronics |       547.11 |        8 |

    </div>

We can also use numerical filtering, as you can see in the next example, where we filter for sales amounts greater than $500.

=== "SQL"

    When it comes to numerical filtering in SQL, the process is similar to the previous example, where we use the `WHERE` clause to filter rows based on a given condition, but here we use a numerical value instead of a string value. In the below example, we filter for sales amounts greater than `500`.

    ```py {.sql linenums="1" title="Filter for high value transactions"}
    high_value_sales_txt: str = """
        SELECT *
        FROM sales
        WHERE sales_amount > 500
    """
    high_value_sales_sql: pd.DataFrame = pd.read_sql(high_value_sales_txt, conn)
    print(f"Number of high-value Sales: {len(high_value_sales_sql)}")
    print(pd.read_sql(high_value_sales_txt + "LIMIT 5", conn))
    print(pd.read_sql(high_value_sales_txt + "LIMIT 5", conn).to_markdown())
    ```

    <div class="result" markdown>

    ```txt
    Number of high-value Sales: 43
    ```

    ```txt
                      date  customer_id  product_id     category  sales_amount  quantity
    0  2023-01-03 00:00:00           15          29         Home        994.51         5
    1  2023-01-09 00:00:00           75           9  Electronics        746.73         2
    2  2023-01-10 00:00:00           75          24        Books        723.73         6
    3  2023-01-12 00:00:00           24          44  Electronics        547.11         8
    4  2023-01-13 00:00:00            3           8     Clothing        513.73         5
    ```

    |      | date                | customer_id | product_id | category    | sales_amount | quantity |
    | ---: | :------------------ | ----------: | ---------: | :---------- | -----------: | -------: |
    |    0 | 2023-01-03 00:00:00 |          15 |         29 | Home        |       994.51 |        5 |
    |    1 | 2023-01-09 00:00:00 |          75 |          9 | Electronics |       746.73 |        2 |
    |    2 | 2023-01-10 00:00:00 |          75 |         24 | Books       |       723.73 |        6 |
    |    3 | 2023-01-12 00:00:00 |          24 |         44 | Electronics |       547.11 |        8 |
    |    4 | 2023-01-13 00:00:00 |           3 |          8 | Clothing    |       513.73 |        5 |

    </div>

In addition to subsetting a table by rows (aka _filtering_), we can also subset a table by columns (aka _selecting_). This allows us to create a new DataFrame with only the relevant columns we want to work with. This is useful when we want to focus on specific attributes of the data, such as dates, categories, or sales amounts.


=== "SQL"

    To select specific columns in SQL, we can use the `SELECT` statement to specify the columns we want to retrieve from the table. This allows us to create a new DataFrame with only the relevant columns. We can use the [`pd.read_sql()`][pandas-read_sql] function to execute SQL queries and retrieve the data from the database.

    For more information about selecting specific columns in SQL, see the [SQL SELECT statement documentation][sqlite-select].

    ```py {.sql linenums="1" title="Select specific columns"}
    sales_summary_txt: str = """
        SELECT date, category, sales_amount
        FROM sales
    """
    sales_summary_sql: pd.DataFrame = pd.read_sql(sales_summary_txt, conn)
    print(f"Selected columns in Sales: {len(sales_summary_sql)}")
    print(pd.read_sql(sales_summary_txt + "LIMIT 5", conn))
    print(pd.read_sql(sales_summary_txt + "LIMIT 5", conn).to_markdown())
    ```

    <div class="result" markdown>

    ```txt
    Selected columns in Sales: 100
    ```

    ```txt
                      date     category  sales_amount
    0  2023-01-01 00:00:00         Food        490.76
    1  2023-01-02 00:00:00  Electronics        453.94
    2  2023-01-03 00:00:00         Home        994.51
    3  2023-01-04 00:00:00  Electronics        184.17
    4  2023-01-05 00:00:00         Food         27.89
    ```

    |      | date                | category    | sales_amount |
    | ---: | :------------------ | :---------- | -----------: |
    |    0 | 2023-01-01 00:00:00 | Food        |       490.76 |
    |    1 | 2023-01-02 00:00:00 | Electronics |       453.94 |
    |    2 | 2023-01-03 00:00:00 | Home        |       994.51 |
    |    3 | 2023-01-04 00:00:00 | Electronics |       184.17 |
    |    4 | 2023-01-05 00:00:00 | Food        |        27.89 |

    </div>

## 2. Grouping and Aggregation

The second section will cover grouping and aggregation techniques. These operations are essential for summarizing data and extracting insights from large datasets.

=== "SQL"

    In SQL, we can use the aggregate functions like `SUM()`, `AVG()`, `MIN()`, `MAX()`, and `COUNT()` to perform aggregation operations on tables.

    Note here that we are _not_ using the `GROUP BY` clause, which is typically used to group rows that have the same values in specified columns into summary rows. Instead, we are performing a basic aggregation on the entire table.

    ```py {.sql linenums="1" title="Basic aggregation"}
    sales_stats_txt: str = """
        SELECT
            SUM(sales_amount) AS sales_sum,
            AVG(sales_amount) AS sales_mean,
            MIN(sales_amount) AS sales_min,
            MAX(sales_amount) AS sales_max,
            COUNT(*) AS sales_count,
            SUM(quantity) AS quantity_sum,
            AVG(quantity) AS quantity_mean,
            MIN(quantity) AS quantity_min,
            MAX(quantity) AS quantity_max
        FROM sales
    """
    print(f"Sales Statistics: {len(pd.read_sql(sales_stats_txt, conn))}")
    print(pd.read_sql(sales_stats_txt, conn))
    print(pd.read_sql(sales_stats_txt, conn).to_markdown())
    ```

    <div class="result" markdown>

    ```txt
    Sales Statistics: 1
    ```

    ```txt
       sales_sum  sales_mean  sales_min  sales_max  sales_count  quantity_sum  quantity_mean  quantity_min  quantity_max
    0   48227.05    482.2705      15.13     994.61          100           464           4.64             1             9
    ```

    |      | sales_sum | sales_mean | sales_min | sales_max | sales_count | quantity_sum | quantity_mean | quantity_min | quantity_max |
    | ---: | --------: | ---------: | --------: | --------: | ----------: | -----------: | ------------: | -----------: | -----------: |
    |    0 |   48227.1 |    482.271 |     15.13 |    994.61 |         100 |          464 |          4.64 |            1 |            9 |

    </div>

It is also possible to group the data by a specific column and then apply aggregation functions to summarize the data by group.

=== "SQL"

    In SQL, we can use the `GROUP BY` clause to group rows that have the same values in specified columns into summary rows. We can then apply aggregate functions like `SUM()`, `AVG()`, and `COUNT()` in the `SELECT` clause to summarize the data by group.

    ```py {.sql linenums="1" title="Group by category and aggregate"}
    category_sales_txt: str = """
        SELECT
            category,
            SUM(sales_amount) AS total_sales,
            AVG(sales_amount) AS average_sales,
            COUNT(*) AS transaction_count,
            SUM(quantity) AS total_quantity
        FROM sales
        GROUP BY category
    """
    print(f"Category Sales Summary: {len(pd.read_sql(category_sales_txt, conn))}")
    print(pd.read_sql(category_sales_txt + "LIMIT 5", conn))
    print(pd.read_sql(category_sales_txt + "LIMIT 5", conn).to_markdown())
    ```

    <div class="result" markdown>

    ```txt
    Category Sales Summary: 5
    ```

    ```txt
          category  total_sales  average_sales  transaction_count  total_quantity
    0        Books     10154.83     441.514348                 23             100
    1     Clothing      7325.31     457.831875                 16              62
    2  Electronics     11407.45     407.408929                 28             147
    3         Food     12995.57     541.482083                 24             115
    4         Home      6343.89     704.876667                  9              40
    ```

    |      | category    | total_sales | average_sales | transaction_count | total_quantity |
    | ---: | :---------- | ----------: | ------------: | ----------------: | -------------: |
    |    0 | Books       |     10154.8 |       441.514 |                23 |            100 |
    |    1 | Clothing    |     7325.31 |       457.832 |                16 |             62 |
    |    2 | Electronics |     11407.5 |       407.409 |                28 |            147 |
    |    3 | Food        |     12995.6 |       541.482 |                24 |            115 |
    |    4 | Home        |     6343.89 |       704.877 |                 9 |             40 |

    </div>

We can rename the columns for clarity by simply assigning new names.

=== "SQL"

    In SQL, we can use the `AS` keyword to rename columns in the `SELECT` clause. This allows us to provide more descriptive names for the aggregated columns.

    In this example, we provide the same aggregation as before, but from within a subquery. Then, in the parent query, we rename the columns for clarity.

    ```py {.sql linenums="1" title="Rename columns for clarity"}
    category_sales_renamed_txt: str = """
        SELECT
            category,
            total_sales AS `Total Sales`,
            average_sales AS `Average Sales`,
            transaction_count AS `Transaction Count`,
            total_quantity AS `Total Quantity`
        FROM (
            SELECT
                category,
                SUM(sales_amount) AS total_sales,
                AVG(sales_amount) AS average_sales,
                COUNT(*) AS transaction_count,
                SUM(quantity) AS total_quantity
            FROM sales
            GROUP BY category
        ) AS sales_summary
    """
    print(f"Renamed Category Sales Summary: {len(pd.read_sql(category_sales_renamed_txt, conn))}")
    print(pd.read_sql(category_sales_renamed_txt + "LIMIT 5", conn))
    print(pd.read_sql(category_sales_renamed_txt + "LIMIT 5", conn).to_markdown())
    ```

    <div class="result" markdown>

    ```txt
    Renamed Category Sales Summary: 5
    ```

    ```txt
                 Total Sales  Average Sales  Transaction Count  Total Quantity
    category
    Books           10154.83     441.514348                 23             100
    Clothing         7325.31     457.831875                 16              62
    Electronics     11407.45     407.408929                 28             147
    Food            12995.57     541.482083                 24             115
    Home             6343.89     704.876667                  9              40
    ```

    | category    | Total Sales | Average Sales | Transaction Count | Total Quantity |
    | :---------- | ----------: | ------------: | ----------------: | -------------: |
    | Books       |     10154.8 |       441.514 |                23 |            100 |
    | Clothing    |     7325.31 |       457.832 |                16 |             62 |
    | Electronics |     11407.5 |       407.409 |                28 |            147 |
    | Food        |     12995.6 |       541.482 |                24 |            115 |
    | Home        |     6343.89 |       704.877 |                 9 |             40 |

    </div>

Having aggregated the data, we can now visualize the results using [Plotly][plotly]. This allows us to create interactive visualizations that can help us better understand the data. The simplest way to do this is to use the [Plotly Express][plotly-express] module, which provides a high-level interface for creating visualizations. Here, we have utilised the [`px.bar()`][plotly-bar] function to create a bar chart of the total sales by category.

=== "SQL"

    The Plotly [`px.bar()`][plotly-bar] function can also receive a Pandas DataFrame, so we can use the results of the SQL query directly. Since the method we are using already returns the group labels in an individual column, we can use that directly in Plotly as the labels for the x-axis.

    ```py {.sql linenums="1" title="Plot the results"}
    fig: go.Figure = px.bar(
        data_frame=pd.read_sql(category_sales_renamed_txt, conn),
        x="category",
        y="Total Sales",
        title="Total Sales by Category",
        text="Transaction Count",
        labels={"Total Sales": "Total Sales ($)", "category": "Product Category"},
    )
    fig.write_html("images/pt2_total_sales_by_category_sql.html", include_plotlyjs="cdn", full_html=True)
    fig.show()
    ```

    <div class="result" markdown>

    --8<-- "docs/guides/querying-data/images/pt2_total_sales_by_category_sql.html"

    </div>

--8<-- "docs/guides/querying-data/images/pt2_total_sales_by_category_pd.html"
--8<-- "docs/guides/querying-data/images/pt2_total_sales_by_category_sql.html"
--8<-- "docs/guides/querying-data/images/pt2_total_sales_by_category_ps.html"
--8<-- "docs/guides/querying-data/images/pt2_total_sales_by_category_pl.html"


## 3. Joining

The third section will demonstrate how to join DataFrames to combine data from different sources. This is a common operation in data analysis, allowing us to enrich our data with additional information.

Here, we will join the `sales` DataFrame with the `product` DataFrame to get additional information about the products sold.

=== "SQL"

    In SQL, we can use the [`JOIN`][sqlite-tutorial-join] clause to combine rows from two or more tables based on a related column between them. In this case, we will join the `sales` table with the `product` table on the `product_id` column.

    ```py {.sql linenums="1" title="Join sales with product data"}
    sales_with_product_txt: str = """
        SELECT s.*, p.product_name, p.price
        FROM sales s
        LEFT JOIN product p ON s.product_id = p.product_id
    """
    print(f"Sales with Product Information: {len(pd.read_sql(sales_with_product_txt, conn))}")
    print(pd.read_sql(sales_with_product_txt + "LIMIT 5", conn))
    print(pd.read_sql(sales_with_product_txt + "LIMIT 5", conn).to_markdown())
    ```

    <div class="result" markdown>

    ```txt
    Sales with Product Information: 100
    ```

    ```txt
                      date  customer_id  product_id     category  sales_amount  quantity  product_name   price
    0  2023-01-01 00:00:00           52          45         Food        490.76         7    Product 45  493.14
    1  2023-01-02 00:00:00           93          41  Electronics        453.94         5    Product 41  193.39
    2  2023-01-03 00:00:00           15          29         Home        994.51         5    Product 29   80.07
    3  2023-01-04 00:00:00           72          15  Electronics        184.17         7    Product 15  153.67
    4  2023-01-05 00:00:00           61          45         Food         27.89         9    Product 45  493.14
    ```

    |      | date                | customer_id | product_id | category    | sales_amount | quantity | product_name |  price |
    | ---: | :------------------ | ----------: | ---------: | :---------- | -----------: | -------: | :----------- | -----: |
    |    0 | 2023-01-01 00:00:00 |          52 |         45 | Food        |       490.76 |        7 | Product 45   | 493.14 |
    |    1 | 2023-01-02 00:00:00 |          93 |         41 | Electronics |       453.94 |        5 | Product 41   | 193.39 |
    |    2 | 2023-01-03 00:00:00 |          15 |         29 | Home        |       994.51 |        5 | Product 29   |  80.07 |
    |    3 | 2023-01-04 00:00:00 |          72 |         15 | Electronics |       184.17 |        7 | Product 15   | 153.67 |
    |    4 | 2023-01-05 00:00:00 |          61 |         45 | Food        |        27.89 |        9 | Product 45   | 493.14 |

    </div>

In the next step, we will join the resulting DataFrame with the `customer` DataFrame to get customer information for each sale. This allows us to create a complete view of the sales data, including product and customer details.

=== "SQL"

    This process is similar to the previous step, but now we will extend the `sales_with_product` DataFrame to join it with the `customer` DataFrame on the `customer_id` column. This will give us a complete view of the sales data, including product and customer details.

    ```py {.sql linenums="1" title="Join with customer information to get a complete view"}
    complete_sales_txt: str = """
        SELECT
            s.*,
            p.product_name,
            p.price,
            c.customer_name,
            c.city,
            c.state
        FROM sales s
        LEFT JOIN product p ON s.product_id = p.product_id
        LEFT JOIN customer c ON s.customer_id = c.customer_id
    """
    print(f"Complete Sales Data with Customer Information: {len(pd.read_sql(complete_sales_txt, conn))}")
    print(pd.read_sql(complete_sales_txt + "LIMIT 5", conn))
    print(pd.read_sql(complete_sales_txt + "LIMIT 5", conn).to_markdown())
    ```

    <div class="result" markdown>

    ```txt
    Complete Sales Data with Customer Information: 100
    ```

    ```txt
                      date  customer_id  product_id     category  sales_amount  quantity  product_name   price  customer_name      city  state
    0  2023-01-01 00:00:00           52          45         Food        490.76         7    Product 45  493.14    Customer 52   Phoenix     TX
    1  2023-01-02 00:00:00           93          41  Electronics        453.94         5    Product 41  193.39    Customer 93  New York     TX
    2  2023-01-03 00:00:00           15          29         Home        994.51         5    Product 29   80.07    Customer 15  New York     CA
    3  2023-01-04 00:00:00           72          15  Electronics        184.17         7    Product 15  153.67    Customer 72   Houston     IL
    4  2023-01-05 00:00:00           61          45         Food         27.89         9    Product 45  493.14    Customer 61   Phoenix     IL
    ```

    |      | date                | customer_id | product_id | category    | sales_amount | quantity | product_name |  price | customer_name | city     | state |
    | ---: | :------------------ | ----------: | ---------: | :---------- | -----------: | -------: | :----------- | -----: | :------------ | :------- | :---- |
    |    0 | 2023-01-01 00:00:00 |          52 |         45 | Food        |       490.76 |        7 | Product 45   | 493.14 | Customer 52   | Phoenix  | TX    |
    |    1 | 2023-01-02 00:00:00 |          93 |         41 | Electronics |       453.94 |        5 | Product 41   | 193.39 | Customer 93   | New York | TX    |
    |    2 | 2023-01-03 00:00:00 |          15 |         29 | Home        |       994.51 |        5 | Product 29   |  80.07 | Customer 15   | New York | CA    |
    |    3 | 2023-01-04 00:00:00 |          72 |         15 | Electronics |       184.17 |        7 | Product 15   | 153.67 | Customer 72   | Houston  | IL    |
    |    4 | 2023-01-05 00:00:00 |          61 |         45 | Food        |        27.89 |        9 | Product 45   | 493.14 | Customer 61   | Phoenix  | IL    |

    </div>

Once we have the complete sales data, we can calculate the revenue for each sale by multiplying the price and quantity (columns from different tables). We can also compare this calculated revenue with the sales amount to identify any discrepancies.

=== "SQL"

    In SQL, we can calculate the revenue for each sale by multiplying the `price` and `quantity` columns. We can then compare this calculated revenue with the `sales_amount` column to identify any discrepancies.

    ```py {.sql linenums="1" title="Calculate revenue and compare with sales amount"}
    revenue_comparison_txt: str = """
        SELECT
            s.sales_amount,
            p.price,
            s.quantity,
            (p.price * s.quantity) AS calculated_revenue,
            (s.sales_amount - (p.price * s.quantity)) AS price_difference
        FROM sales s
        LEFT JOIN product p ON s.product_id = p.product_id
    """
    print(
        f"Complete Sales Data with Calculated Revenue and Price Difference: {len(pd.read_sql(revenue_comparison_txt, conn))}"
    )
    print(pd.read_sql(revenue_comparison_txt + "LIMIT 5", conn))
    print(pd.read_sql(revenue_comparison_txt + "LIMIT 5", conn).to_markdown())
    ```

    <div class="result" markdown>

    ```txt
    Complete Sales Data with Calculated Revenue and Price Difference: 100
    ```

    ```txt
       sales_amount   price  quantity  calculated_revenue  price_difference
    0        490.76  493.14         7             3451.98          -2961.22
    1        453.94  193.39         5              966.95           -513.01
    2        994.51   80.07         5              400.35            594.16
    3        184.17  153.67         7             1075.69           -891.52
    4         27.89  493.14         9             4438.26          -4410.37
    ```

    |      | sales_amount |  price | quantity | calculated_revenue | price_difference |
    | ---: | -----------: | -----: | -------: | -----------------: | ---------------: |
    |    0 |       490.76 | 493.14 |        7 |            3451.98 |         -2961.22 |
    |    1 |       453.94 | 193.39 |        5 |             966.95 |          -513.01 |
    |    2 |       994.51 |  80.07 |        5 |             400.35 |           594.16 |
    |    3 |       184.17 | 153.67 |        7 |            1075.69 |          -891.52 |
    |    4 |        27.89 | 493.14 |        9 |            4438.26 |         -4410.37 |

    </div>

## 4. Window Functions

Window functions are a powerful feature in Pandas that allow us to perform calculations across a set of rows related to the current row. This is particularly useful for time series data, where we may want to calculate rolling averages, cumulative sums, or other metrics based on previous or subsequent rows.

To understand more about the nuances of the window functions, check out some of these guides:

- [Analyzing data with window functions][analysing-window-functions]
- [SQL Window Functions Visualized][visualising-window-functions]

In this section, we will demonstrate how to use window functions to analyze sales data over time. We will start by converting the `date` column to a datetime type, which is necessary for time-based calculations. We will then group the data by date and calculate the total sales for each day.

The first thing that we will do is to group the sales data by date and calculate the total sales for each day. This will give us a daily summary of sales, which we can then use to analyze trends over time.

=== "SQL"

    In SQL, we can use the `GROUP BY` clause to group the data by the `date` column and then use the `SUM()` function to calculate the total sales for each day. This will give us a daily summary of sales, which we can then use to analyze trends over time.

    ```py {.sql linenums="1" title="Time-based window function"}
    daily_sales_txt: str = """
        SELECT
            date,
            SUM(sales_amount) AS total_sales
        FROM sales
        GROUP BY date
        ORDER BY date
    """
    print(f"Daily Sales Summary: {len(pd.read_sql(daily_sales_txt, conn))}")
    print(pd.read_sql(daily_sales_txt + "LIMIT 5", conn))
    print(pd.read_sql(daily_sales_txt + "LIMIT 5", conn).to_markdown())
    ```

    <div class="result" markdown>

    ```txt
    Daily Sales Summary: 100
    ```

    ```txt
                      date  total_sales
    0  2023-01-01 00:00:00       490.76
    1  2023-01-02 00:00:00       453.94
    2  2023-01-03 00:00:00       994.51
    3  2023-01-04 00:00:00       184.17
    4  2023-01-05 00:00:00        27.89
    ```

    |      | date                | total_sales |
    | ---: | :------------------ | ----------: |
    |    0 | 2023-01-01 00:00:00 |      490.76 |
    |    1 | 2023-01-02 00:00:00 |      453.94 |
    |    2 | 2023-01-03 00:00:00 |      994.51 |
    |    3 | 2023-01-04 00:00:00 |      184.17 |
    |    4 | 2023-01-05 00:00:00 |       27.89 |

    </div>

Next, we will calculate the lag and lead values for the sales amount. This allows us to compare the current day's sales with the previous and next days' sales.

=== "SQL"

    In SQL, we can use the `LAG()` and `LEAD()` window functions to calculate the lag and lead values for the sales amount. These functions allow us to access data from previous and next rows in the result set without needing to join the table to itself.

    The part that is important to note here is that the `LAG()` and `LEAD()` functions are used in conjunction with the `OVER` clause, which defines the window over which the function operates. In this case, we are ordering by the `date` column to ensure that the lag and lead values are calculated based on the chronological order of the sales data.

    ```py {.sql linenums="1" title="Calculate lag and lead"}
    lag_lead_txt: str = """
        SELECT
            date AS sale_date,
            SUM(sales_amount) AS total_sales,
            LAG(SUM(sales_amount)) OVER (ORDER BY date) AS previous_day_sales,
            LEAD(SUM(sales_amount)) OVER (ORDER BY date) AS next_day_sales
        FROM sales
        GROUP BY date
        ORDER BY date
    """
    lag_lead_df_sql: pd.DataFrame = pd.read_sql(lag_lead_txt, conn)
    print(f"Daily Sales with Lag and Lead: {len(lag_lead_df_sql)}")
    print(pd.read_sql(lag_lead_txt + " LIMIT 5", conn))
    print(pd.read_sql(lag_lead_txt + " LIMIT 5", conn).to_markdown())
    ```

    <div class="result" markdown>

    ```txt
    Daily Sales with Lag and Lead: 100
    ```

    ```txt
                 sale_date  total_sales  previous_day_sales  next_day_sales
    0  2023-01-01 00:00:00        490.76                 NaN          453.94
    1  2023-01-02 00:00:00        453.94              490.76          994.51
    2  2023-01-03 00:00:00        994.51              453.94          184.17
    3  2023-01-04 00:00:00        184.17              994.51           27.89
    4  2023-01-05 00:00:00         27.89              184.17          498.95
    ```

    |      | sale_date           | total_sales | previous_day_sales | next_day_sales |
    | ---: | :------------------ | ----------: | -----------------: | -------------: |
    |    0 | 2023-01-01 00:00:00 |      490.76 |                nan |         453.94 |
    |    1 | 2023-01-02 00:00:00 |      453.94 |             490.76 |         994.51 |
    |    2 | 2023-01-03 00:00:00 |      994.51 |             453.94 |         184.17 |
    |    3 | 2023-01-04 00:00:00 |      184.17 |             994.51 |          27.89 |
    |    4 | 2023-01-05 00:00:00 |       27.89 |             184.17 |         498.95 |

    </div>

Now, we can calculate the day-over-day change in sales. This is done by subtracting the previous day's sales from the current day's sales. Then secondly, we can calculate the percentage change in sales using the formula:

```txt
((current_day_sales - previous_day_sales) / previous_day_sales) * 100
```

=== "SQL"

    In SQL, we can calculate the day-over-day change in sales by subtracting the `previous_day_sales` column from the `total_sales` column. We can also calculate the percentage change in sales using the formula:

    ```txt
    ((current_day_sales - previous_day_sales) / previous_day_sales) * 100
    ```

    ```py {.sql linenums="1" title="Day-over-day change already calculated"}
    dod_change_txt: str = """
        SELECT
            sale_date,
            total_sales,
            previous_day_sales,
            next_day_sales,
            total_sales - previous_day_sales AS day_over_day_change,
            ((total_sales - previous_day_sales) / previous_day_sales) * 100 AS pct_change
        FROM (
            SELECT
                date AS sale_date,
                SUM(sales_amount) AS total_sales,
                LAG(SUM(sales_amount)) OVER (ORDER BY date) AS previous_day_sales,
                LEAD(SUM(sales_amount)) OVER (ORDER BY date) AS next_day_sales
            FROM sales
            GROUP BY date
        )
        ORDER BY sale_date
    """
    dod_change_df_sql: pd.DataFrame = pd.read_sql(dod_change_txt, conn)
    print(f"Daily Sales with Day-over-Day Change: {len(dod_change_df_sql)}")
    print(dod_change_df_sql.head(5))
    print(dod_change_df_sql.head(5).to_markdown())
    ```

    <div class="result" markdown>

    ```txt
    Daily Sales with Day-over-Day Change: 100
    ```

    ```txt
                 sale_date  total_sales  previous_day_sales  next_day_sales  day_over_day_change  pct_change
    0  2023-01-01 00:00:00       490.76                 NaN          453.94                  NaN         NaN
    1  2023-01-02 00:00:00       453.94              490.76          994.51               -36.82   -7.502649
    2  2023-01-03 00:00:00       994.51              453.94          184.17               540.57  119.084020
    3  2023-01-04 00:00:00       184.17              994.51           27.89              -810.34  -81.481333
    4  2023-01-05 00:00:00        27.89              184.17          498.95              -156.28  -84.856383
    ```

    |      | sale_date           | total_sales | previous_day_sales | next_day_sales | day_over_day_change | pct_change |
    | ---: | :------------------ | ----------: | -----------------: | -------------: | ------------------: | ---------: |
    |    0 | 2023-01-01 00:00:00 |      490.76 |                nan |         453.94 |                 nan |        nan |
    |    1 | 2023-01-02 00:00:00 |      453.94 |             490.76 |         994.51 |              -36.82 |   -7.50265 |
    |    2 | 2023-01-03 00:00:00 |      994.51 |             453.94 |         184.17 |              540.57 |    119.084 |
    |    3 | 2023-01-04 00:00:00 |      184.17 |             994.51 |          27.89 |             -810.34 |   -81.4813 |
    |    4 | 2023-01-05 00:00:00 |       27.89 |             184.17 |         498.95 |             -156.28 |   -84.8564 |

    </div>

Next, we will calculate the rolling average of sales over a 7-day window. Rolling averages (aka moving averages) are useful for smoothing out short-term fluctuations and highlighting longer-term trends in the data. This is particularly useful in time series analysis, where we want to understand the underlying trend in the data without being overly influenced by short-term variations. It is also a very common technique used in financial analysis to analyze stock prices, sales data, and other time series data.

=== "SQL"

    In SQL, we can calculate the 7-day moving average of sales using the `AVG()` window function with the `OVER` clause. It is important to include this `OVER` clause, because it is what the SQL engine uses to determine that it should be a Window function, rather than a regular aggregate function (which is specified using the `GROUP BY` clause).

    Here in our example, there are three different parts to the Window function:

    1. **The `ORDER BY` clause**: This specifies the order of the rows in the window. In this case, we are ordering by the `sale_date` column.
    2. **The `ROWS BETWEEN` clause**: This specifies the range of rows to include in the window. In this case, we are including the number of rows from 6 preceding rows to the current row. This means that for each row, the window will include the current row and the 6 rows before it, giving us a total of 7 rows in the window. It is important that you specify the `ORDER BY` clause before the `ROWS BETWEEN` clause to ensure that the correct rows are included in the window.
    3. **The `AVG()` function**: This calculates the average of the `total_sales` column over the specified window.

    Another perculiarity to note here is around the use of the sub-query. The sub-query is used to first calculate the daily sales, including the previous and next day sales, and the day-over-day change. This is because we need to calculate the moving average over the daily sales, rather than the individual sales transactions. The sub-query allows us to aggregate the sales data by date before calculating the moving average. The only change that we are including in the outer-query is the addition of the moving average calculation.

    ```py {.sql linenums="1" title="Calculate 7-day moving average"}
    rolling_avg_txt: str = """
        SELECT
            sale_date,
            total_sales,
            previous_day_sales,
            next_day_sales,
            day_over_day_change,
            pct_change,
            AVG(total_sales) OVER (ORDER BY sale_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS "7d_moving_avg"
        FROM (
            SELECT
                date AS sale_date,
                SUM(sales_amount) AS total_sales,
                LAG(SUM(sales_amount)) OVER (ORDER BY date) AS previous_day_sales,
                LEAD(SUM(sales_amount)) OVER (ORDER BY date) AS next_day_sales,
                SUM(sales_amount) - LAG(SUM(sales_amount)) OVER (ORDER BY date) AS day_over_day_change,
                (SUM(sales_amount) / NULLIF(LAG(SUM(sales_amount)) OVER (ORDER BY date), 0) - 1) * 100 AS pct_change
            FROM sales
            GROUP BY date
        ) AS daily_sales
        ORDER BY sale_date
    """
    window_df_sql: pd.DataFrame = pd.read_sql(rolling_avg_txt, conn)
    print(f"Daily Sales with 7-Day Moving Average: {len(window_df_sql)}")
    print(pd.read_sql(rolling_avg_txt + " LIMIT 5", conn))
    print(pd.read_sql(rolling_avg_txt + " LIMIT 5", conn).to_markdown())
    ```

    <div class="result" markdown>

    ```txt
    Daily Sales with 7-Day Moving Average: 100
    ```

    ```txt
                 sale_date  total_sales  previous_day_sales  next_day_sales  day_over_day_change  pct_change  7d_moving_avg
    0  2023-01-01 00:00:00       490.76                 NaN          453.94                  NaN         NaN     490.760000
    1  2023-01-02 00:00:00       453.94              490.76          994.51               -36.82   -7.502649     472.350000
    2  2023-01-03 00:00:00       994.51              453.94          184.17               540.57  119.084020     646.403333
    3  2023-01-04 00:00:00       184.17              994.51           27.89              -810.34  -81.481333     530.845000
    4  2023-01-05 00:00:00        27.89              184.17          498.95              -156.28  -84.856383     430.254000
    ```

    |      | sale_date           | total_sales | previous_day_sales | next_day_sales | day_over_day_change | pct_change | 7d_moving_avg |
    | ---: | :------------------ | ----------: | -----------------: | -------------: | ------------------: | ---------: | ------------: |
    |    0 | 2023-01-01 00:00:00 |      490.76 |                nan |         453.94 |                 nan |        nan |        490.76 |
    |    1 | 2023-01-02 00:00:00 |      453.94 |             490.76 |         994.51 |              -36.82 |   -7.50265 |        472.35 |
    |    2 | 2023-01-03 00:00:00 |      994.51 |             453.94 |         184.17 |              540.57 |    119.084 |       646.403 |
    |    3 | 2023-01-04 00:00:00 |      184.17 |             994.51 |          27.89 |             -810.34 |   -81.4813 |       530.845 |
    |    4 | 2023-01-05 00:00:00 |       27.89 |             184.17 |         498.95 |             -156.28 |   -84.8564 |       430.254 |

    </div>

Finally, we can visualize the daily sales data along with the 7-day moving average using Plotly. This allows us to see the trends in sales over time and how the moving average smooths out the fluctuations in daily sales.

For this, we will again utilise [Plotly][plotly] to create an interactive line chart that displays both the daily sales and the 7-day moving average. The chart will have the date on the x-axis and the sales amount on the y-axis, with two lines representing the daily sales and the moving average.

The graph will be [instantiated][python-class-instantiation] using the [`go.Figure()`][plotly-figure] class, and using the [`.add_trace()`][plotly-add-traces] method we will add two traces to the figure: one for the daily sales and one for the 7-day moving average. The [`go.Scatter()`][plotly-scatter] class is used to create the line traces, by defining `mode="lines"` to display the data as a line chart.

Finally, we will use the [`.update_layout()`][plotly-update_layout] method to set the titles for the chart, and the position of the legend.

=== "SQL"

    Plotly is easily able to handle Pandas DataFrames, so we can directly parse the columns from the DataFrame to create the traces for the daily sales and the 7-day moving average.

    ```py {.sql linenums="1" title="Plot results"}
    fig: go.Figure = (
        go.Figure()
        .add_trace(
            go.Scatter(
                x=window_df_sql["sale_date"],
                y=window_df_sql["total_sales"],
                mode="lines",
                name="Daily Sales",
            )
        )
        .add_trace(
            go.Scatter(
                x=window_df_sql["sale_date"],
                y=window_df_sql["7d_moving_avg"],
                mode="lines",
                name="7-Day Moving Average",
                line_width=3,
            )
        )
        .update_layout(
            title="Daily Sales with 7-Day Moving Average",
            xaxis_title="Date",
            yaxis_title="Sales Amount ($)",
            legend_orientation="h",
            legend_yanchor="bottom",
            legend_y=1,
        )
    )
    fig.write_html("images/pt4_daily_sales_with_7d_avg_sql.html", include_plotlyjs="cdn", full_html=True)
    fig.show()
    ```

    <div class="result" markdown>

    --8<-- "docs/guides/querying-data/images/pt4_daily_sales_with_7d_avg_sql.html"

    </div>

--8<-- "docs/guides/querying-data/images/pt4_daily_sales_with_7d_avg_pd.html"
--8<-- "docs/guides/querying-data/images/pt4_daily_sales_with_7d_avg_sql.html"
--8<-- "docs/guides/querying-data/images/pt4_daily_sales_with_7d_avg_ps.html"
--8<-- "docs/guides/querying-data/images/pt4_daily_sales_with_7d_avg_pl.html"


## 5. Ranking and Partitioning

The fifth section will demonstrate how to rank and partition data. This is useful for identifying top performers, such as the highest spending customers or the most popular products.

=== "SQL"

    In SQL, we can use the `DENSE_RANK()` window function to rank values in a query. This function assigns a rank to each row within a partition of a given result set, with no gaps in the ranking values. Note that this function can only be used in congunction with the `OVER` clause, which defines it as a Window function. The `ORDER BY` clause within the `OVER` clause specifies the order in which the rows are ranked.

    ```py {.sql linenums="1" title="Rank customers by total spending"}
    customer_spending_txt: str = """
        SELECT
            customer_id,
            SUM(sales_amount) AS total_spending,
            DENSE_RANK() OVER (ORDER BY SUM(sales_amount) DESC) AS rank
        FROM sales
        GROUP BY customer_id
        ORDER BY rank
    """
    customer_spending_sql: pd.DataFrame = pd.read_sql(customer_spending_txt, conn)
    print(f"Customer Spending Summary: {len(customer_spending_sql)}")
    print(customer_spending_sql.head(5))
    print(customer_spending_sql.head(5).to_markdown())
    ```

    <div class="result" markdown>

    ```txt
    Customer Spending Summary: 61
    ```

    ```txt
       customer_id  total_spending  rank
    0           15         2297.55     1
    1            4         2237.49     2
    2           62         2177.35     3
    3           60         2086.09     4
    4           21         2016.95     5
    ```

    |      | customer_id | total_spending | rank |
    | ---: | ----------: | -------------: | ---: |
    |    0 |          15 |        2297.55 |    1 |
    |    1 |           4 |        2237.49 |    2 |
    |    2 |          62 |        2177.35 |    3 |
    |    3 |          60 |        2086.09 |    4 |
    |    4 |          21 |        2016.95 |    5 |

    </div>

Next, we will rank products based on the quantity sold, partitioned by the product category. This will help us identify the most popular products within each category.

=== "SQL"

    In SQL, we can use the `RANK()` window function to rank products within each category based on the total quantity sold. The `PARTITION BY` clause allows us to partition the data by `category`, and the `ORDER BY` clause specifies the order in which the rows are ranked within each partition.

    ```py {.sql linenums="1" title="Rank products by quantity sold, by category"}
    product_popularity_txt: str = """
        SELECT
            category,
            product_id,
            SUM(quantity) AS total_quantity,
            RANK() OVER (PARTITION BY category ORDER BY SUM(quantity) DESC) AS rank
        FROM sales
        GROUP BY category, product_id
        ORDER BY rank
    """
    print(f"Product Popularity: {len(pd.read_sql(product_popularity_txt, conn))}")
    print(pd.read_sql(product_popularity_txt + "LIMIT 10", conn))
    print(pd.read_sql(product_popularity_txt + "LIMIT 10", conn).to_markdown())
    ```

    <div class="result" markdown>

    ```txt
    Product Popularity: 78
    ```

    ```txt
          category  product_id  total_quantity  rank
    0        Books          11              14     1
    1     Clothing           7               9     1
    2  Electronics          37              16     1
    3         Food          45              34     1
    4         Home           3              10     1
    5        Books          28               9     2
    6     Clothing          35               8     2
    7  Electronics          35              11     2
    8         Food           1              16     2
    9         Home          48               5     2
    ```

    |      | category    | product_id | total_quantity | rank |
    | ---: | :---------- | ---------: | -------------: | ---: |
    |    0 | Books       |         11 |             14 |    1 |
    |    1 | Clothing    |          7 |              9 |    1 |
    |    2 | Electronics |         37 |             16 |    1 |
    |    3 | Food        |         45 |             34 |    1 |
    |    4 | Home        |          3 |             10 |    1 |
    |    5 | Books       |         28 |              9 |    2 |
    |    6 | Clothing    |         35 |              8 |    2 |
    |    7 | Electronics |         35 |             11 |    2 |
    |    8 | Food        |          1 |             16 |    2 |
    |    9 | Home        |         48 |              5 |    2 |

    </div>

## Conclusion

This comprehensive guide has demonstrated how to perform essential data querying and manipulation operations across four powerful tools: [Pandas], [SQL][sqlite], [PySpark], and [Polars]. Each tool brings unique advantages to the data processing landscape, and understanding their strengths helps you choose the right tool for your specific use case.


### Tool Comparison and Use Cases

<div class="grid cards" markdown>

-   [**Pandas**][pandas] has an extensive ecosystem, making it ideal for:

    - Small to medium datasets (up to millions of rows)
    - Interactive data exploration and visualization
    - Data preprocessing for machine learning workflows
    - Quick statistical analysis and reporting

    ---

    **Pandas** remains the go-to choice for exploratory data analysis and rapid prototyping.

-   [**SQL**][sqlite] excels in:

    - Working with relational databases and data warehouses
    - Complex joins and subqueries
    - Declarative data transformations
    - Team environments where SQL knowledge is widespread

    ---

    **SQL** provides the universal language of data with unmatched expressiveness for complex queries

-   [**PySpark**][pyspark] is great for when you need:

    - Processing datasets that don't fit in memory (terabytes or larger)
    - Distributed computing across clusters
    - Integration with Hadoop ecosystem components
    - Scalable machine learning with MLlib

    ---

    **PySpark** unlocks the power of distributed computing for big data scenarios.

-   [**Polars**][polars] is particularly valuable for:

    - Large datasets that require fast processing (gigabytes to small terabytes)
    - Performance-critical applications
    - Memory-constrained environments
    - Lazy evaluation and query optimization

    ---

    **Polars** emerges as the high-performance alternative with excellent memory efficiency.

</div>


### Key Techniques Covered

Throughout this guide, we've explored fundamental data manipulation patterns that remain consistent across all tools:

1. **Data Filtering and Selection** - Essential for subsetting data based on conditions
2. **Grouping and Aggregation** - Critical for summarizing data by categories
3. **Joining and Merging** - Necessary for combining data from multiple sources
4. **Window Functions** - Powerful for time-series analysis and advanced calculations
5. **Ranking and Partitioning** - Useful for identifying top performers and comparative analysis


### Best Practices and Recommendations

When working with any of these tools, consider these best practices:

- **Start with the right tool**: Match your tool choice to your data size, infrastructure, and team expertise
- **Understand your data**: Always examine data types, null values, and distributions before processing
- **Optimize for readability**: Write clear, well-documented code that your future self and teammates can understand
- **Profile performance**: Measure execution time and memory usage, especially for large datasets
- **Leverage built-in optimizations**: Use vectorized operations, avoid loops, and take advantage of lazy evaluation where available


### Moving Forward

The data landscape continues to evolve rapidly, with new tools and techniques emerging regularly. The fundamental concepts demonstrated in this guide—filtering, grouping, joining, and analytical functions—remain constant across platforms. By mastering these core concepts, you'll be well-equipped to adapt to new tools and technologies as they arise.

Whether you're analyzing customer behavior, processing sensor data, or building machine learning models, the techniques in this guide provide a solid foundation for effective data manipulation. Remember that the best tool is often the one that best fits your specific requirements for performance, scalability, and team capabilities.

Continue practicing with real datasets, explore advanced features of each tool, and stay curious about emerging technologies in the data processing ecosystem. The skills you've learned here will serve as building blocks for increasingly sophisticated data analysis and engineering tasks.


<!--
-- ------------------------ --
--  Shortcuts & Hyperlinks  --
-- ------------------------ --
-->

<!-- Python -->
[python-print]: https://docs.python.org/3/library/functions.html#print
[python-class-instantiation]: https://docs.python.org/3/tutorial/classes.html#:~:text=example%20class%22.-,Class%20instantiation,-uses%20function%20notation
[numpy]: https://numpy.org/

<!-- Storage -->
[hdfs]: https://hadoop.apache.org/docs/r1.2.1/hdfs_design.html
[s3]: https://aws.amazon.com/s3/
[adls]: https://learn.microsoft.com/en-us/azure/storage/blobs/data-lake-storage-introduction
[jdbc]: https://spark.apache.org/docs/latest/sql-data-sources-jdbc.html

<!-- Guides -->
[analysing-window-functions]: https://docs.snowflake.com/en/user-guide/functions-window-using
[visualising-window-functions]: https://medium.com/learning-sql/sql-window-function-visualized-fff1927f00f2
[querying-data]: https://github.com/data-science-extensions/data-science-extensions/blob/main/docs/guides/querying-data/index.md

<!-- Pandas -->
[pandas]: https://pandas.pydata.org/
[pandas-head]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html
[pandas-read_sql]: https://pandas.pydata.org/docs/reference/api/pandas.read_sql.html
[pandas-subsetting]: https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html
[pandas-agg]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.agg.html
[pandas-groupby]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html
[pandas-groupby-agg]: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.agg.html
[pandas-groupby-sum]: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.sum.html
[pandas-columns]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.columns.html
[pandas-rename]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html
[pandas-reset_index]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.reset_index.html
[pandas-merge]: https://pandas.pydata.org/docs/reference/api/pandas.merge.html
[pandas-shift]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shift.html
[pandas-sort_values]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html
[pandas-pct_change]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pct_change.html
[pandas-rolling]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html
[pandas-rolling-mean]: https://pandas.pydata.org/docs/reference/api/pandas.core.window.rolling.Rolling.mean.html
[pandas-rank]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rank.html
[pandas-assign]: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.assign.html

<!-- SQL -->
[postgresql]: https://www.postgresql.org/
[mysql]: https://www.mysql.com/
[t-sql]: https://learn.microsoft.com/en-us/sql/t-sql/
[pl-sql]: https://www.oracle.com/au/database/technologies/appdev/plsql.html
[sql-wiki]: https://en.wikipedia.org/wiki/SQL
[sql-iso]: https://www.iso.org/standard/76583.html
[sqlite]: https://sqlite.org/
[sqlite3]: https://docs.python.org/3/library/sqlite3.html
[sqlite3-connect]: https://docs.python.org/3/library/sqlite3.html#sqlite3.connect
[sqlite-where]: https://sqlite.org/lang_select.html#whereclause
[sqlite-select]: https://sqlite.org/lang_select.html
[sqlite-tutorial-join]: https://www.sqlitetutorial.net/sqlite-join/

<!-- SQL -->
[spark-sql]: https://spark.apache.org/sql/
[pyspark]: https://spark.apache.org/docs/latest/api/python/
[pyspark-sparksession]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.SparkSession.html
[pyspark-builder]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.SparkSession.html#pyspark.sql.SparkSession.builder
[pyspark-getorcreate]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.SparkSession.builder.getOrCreate.html
[pyspark-createdataframe]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.SparkSession.createDataFrame.html
[pyspark-show]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.show.html
[pyspark-create-dataframe-from-dict]: https://sparkbyexamples.com/pyspark/pyspark-create-dataframe-from-dictionary/
[pyspark-filter]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.filter.html
[pyspark-where]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.where.html
[pyspark-filtering]: https://sparkbyexamples.com/pyspark/pyspark-where-filter/
[pyspark-select]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.select.html
[pyspark-agg]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.agg.html
[pyspark-groupby]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.groupBy.html
[pyspark-groupby-agg]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.GroupedData.agg.html
[pyspark-withcolumnsrenamed]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.withColumnsRenamed.html
[pyspark-topandas]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.toPandas.html
[pyspark-join]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.join.html
[pyspark-withcolumns]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.DataFrame.withColumns.html
[pyspark-col]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.col.html
[pyspark-expr]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.expr.html
[pyspark-lead]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.lead.html
[pyspark-lag]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.lag.html
[sparksql-lead]: https://spark.apache.org/docs/latest/api/sql/index.html#lead
[pyspark-window]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.Window.html
[pyspark-avg]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.avg.html
[pyspark-window-orderby]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.Window.orderBy.html
[pyspark-window-partitionby]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.Window.partitionBy.html
[pyspark-window-rowsbetween]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.Window.rowsBetween.html
[pyspark-dense_rank]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.dense_rank.html
[pyspark-desc]: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.desc.html

<!-- Polars -->
[polars]: https://www.pola.rs/
[polars-head]: https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.head.html
[polars-filter]: https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.filter.html
[polars-filtering]: https://docs.pola.rs/user-guide/transformations/time-series/filter/
[polars-col]: https://docs.pola.rs/api/python/stable/reference/expressions/col.html
[polars-select]: https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.select.html
[polars-groupby]: https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.group_by.html
[polars-groupby-agg]: https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.dataframe.group_by.GroupBy.agg.html
[polars-groupby-sum]: https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.dataframe.group_by.GroupBy.sum.html
[polars-rename]: https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.rename.html
[polars-join]: https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.join.html
[polars-with-columns]: https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.with_columns.html
[polars-shift]: https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.shift.html
[polars-sort]: https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.sort.html
[polars-rolling-mean]: https://docs.pola.rs/api/python/dev/reference/expressions/api/polars.Expr.rolling_mean.html
[polars-rank]: https://docs.pola.rs/api/python/dev/reference/expressions/api/polars.Expr.rank.html
[polars-over]: https://docs.pola.rs/api/python/dev/reference/expressions/api/polars.Expr.over.html

<!-- Plotly -->
[plotly]: https://plotly.com/python/
[plotly-express]: https://plotly.com/python/plotly-express/
[plotly-bar]: https://plotly.com/python/bar-charts/
[plotly-figure]: https://plotly.com/python/creating-and-updating-figures/#figures-as-graph-objects
[plotly-add-traces]: https://plotly.com/python/creating-and-updating-figures/#adding-traces
[plotly-scatter]: https://plotly.com/python/line-and-scatter/
[plotly-update_layout]: https://plotly.com/python/reference/layout/
