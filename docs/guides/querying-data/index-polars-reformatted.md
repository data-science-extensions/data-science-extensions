# Data Querying for Data Scientists


### A Comprehensive Guide of using Pandas, SQL, PySpark, and Polars for Data Manipulation Techniques, with Practical Examples and Visualisations


## Introduction

Working as a Data Scientist or Data Engineer often involves querying data from various sources. There are many tools and libraries available to perform these tasks, each with its own strengths and weaknesses. Also, there are many different ways to achieve similar results, depending on the tool or library used. It's important to be familiar with these different methods to choose the best one for your specific use case.

This article provides a comprehensive guide on how to query data using different tools and libraries, including Pandas, SQL, PySpark, and Polars. Each section will cover the setup, data creation, and various querying techniques such as filtering, grouping, joining, window functions, ranking, and sorting. The output will be identical across all tools, but the transformations will be implemented using the specific syntax and features of each library. Therefore allowing you to compare the different approaches and understand the nuances of each method.


## Overview of the Different Libraries

Before we dive into the querying techniques, let's take a moment to understand the different libraries and tools we will be using in this article. Each library has its own strengths and weaknesses, and understanding these can help you choose the right tool for your specific use case.

Throughout this article, you can easily switch between the different libraries by selecting the appropriate tab. Each section will provide the same functionality, but implemented using the specific syntax and features of each library.

### Polars

[Polars] is a fast DataFrame library for Python that is designed for high-performance data manipulation. It is built on top of Rust and provides a DataFrame API similar to Pandas, but with a focus on performance and memory efficiency. Polars is particularly well-suited for large datasets and complex queries.

Polars supports lazy evaluation, which allows for optimizations in query execution. Polars also provides powerful features for filtering, grouping, joining, and aggregating data, making it a great choice for data analysis tasks.

While Polars is a relatively new library compared to Pandas, it has gained popularity for its performance and ease of use. It is designed to be a drop-in replacement for Pandas, allowing users to leverage its performance benefits without significant changes to their existing code. It is particularly useful for data scientists and analysts who need to work with large datasets and require fast data manipulation capabilities. The setup is simple and straightforward, similar to Pandas, and less complex than PySpark. It is a great choice for data analysis tasks that require high performance and memory efficiency.


## Setup

Before we start querying data, we need to set up our environment. This includes importing the necessary libraries, creating sample data, and defining constants that will be used throughout the article. The following sections will guide you through this setup process. The code for this article is also available on GitHub: [querying-data](...) {==**UPDATE URL**==}.

### Polars

```python {.polars linenums="1" title="Setup"}
# StdLib Imports
from typing import Any

# Third Party Imports
import numpy as np
import polars as pl
from plotly import express as px, graph_objects as go, io as pio


# Set seed for reproducibility
np.random.seed(42)

# Determine the number of records to generate
n_records = 100

# Set default Plotly template
pio.templates.default = "simple_white+gridon"

# Set Polars display options
pl.Config.set_tbl_cols(-1)
```

Once the setup is complete, we can proceed to create our sample data. This data will be used for querying and will be consistent across all libraries. All tables will be created from scratch with randomly generated data to simulate a real-world scenario. This is to ensure that the examples are self-contained and can be run without any external dependencies, and also there is no issues about data privacy or security.

For the below data creation steps, we will be defining the tables using Python dictionaries. Each dictionary will represent a table, with keys as column names and values as lists of data. We will then convert these dictionaries into DataFrames or equivalent structures in each library.

First, we will create a sales fact table. This table will contain information about sales transactions, including the date, customer ID, product ID, category, sales amount, and quantity sold.

```python {.python linenums="1" title="Create Sales Fact Data"}
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

```python {.python linenums="1" title="Create Product Dimension Data"}
product_data: dict[str, Any] = {
    "product_id": np.arange(1, 51),
    "product_name": [f"Product {i}" for i in range(1, 51)],
    "price": np.random.uniform(10, 500, 50).round(2),
    "category": np.random.choice(["Electronics", "Clothing", "Food", "Books", "Home"], 50),
    "supplier_id": np.random.randint(1, 10, 50),
}
```

Finally, we will create a customer dimension table. This table will contain information about customers, including the customer ID, name, city, state, and segment.

```python {.python linenums="1" title="Create Customer Dimension Data"}
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

### Polars

To create the dataframes in Polars, we will use the data we generated earlier. We will parse the dictionaries into Polars DataFrames, which will allow us to perform various data manipulation tasks.

```python {.polars linenums="1" title="Create DataFrames"}
df_sales_pl: pl.DataFrame = pl.DataFrame(sales_data)
df_product_pl: pl.DataFrame = pl.DataFrame(product_data)
df_customer_pl: pl.DataFrame = pl.DataFrame(customer_data)
```

Once the data is created, we can check that it has been loaded correctly by displaying the first few rows of each DataFrame. To do this, we will use the [`.head()`][polars-head] method to display the first `5` rows of each DataFrame, and then parse to the [`print()`][python-print] function to display the DataFrame in a readable format.

```python {.polars linenums="1" title="Check Sales DataFrame"}
print(f"Sales DataFrame: {df_sales_pl.shape[0]}")
print(df_sales_pl.head(5))
print(df_sales_pl.head(5).to_pandas().to_markdown())
```

<div class="result" markdown>

```txt
Sales DataFrame: 100
```

```txt
shape: (5, 6)
┌─────────────────────┬─────────────┬────────────┬─────────────┬──────────────┬──────────┐
│ date                ┆ customer_id ┆ product_id ┆ category    ┆ sales_amount ┆ quantity │
│ ---                 ┆ ---         ┆ ---        ┆ ---         ┆ ---          ┆ ---      │
│ datetime[ns]        ┆ i64         ┆ i64        ┆ str         ┆ f64          ┆ i64      │
╞═════════════════════╪═════════════╪════════════╪═════════════╪══════════════╪══════════╡
│ 2023-01-01 00:00:00 ┆ 52          ┆ 45         ┆ Food        ┆ 490.76       ┆ 7        │
│ 2023-01-02 00:00:00 ┆ 93          ┆ 41         ┆ Electronics ┆ 453.94       ┆ 5        │
│ 2023-01-03 00:00:00 ┆ 15          ┆ 29         ┆ Home        ┆ 994.51       ┆ 5        │
│ 2023-01-04 00:00:00 ┆ 72          ┆ 15         ┆ Electronics ┆ 184.17       ┆ 7        │
│ 2023-01-05 00:00:00 ┆ 61          ┆ 45         ┆ Food        ┆ 27.89        ┆ 9        │
└─────────────────────┴─────────────┴────────────┴─────────────┴──────────────┴──────────┘
```

|      | date                | customer_id | product_id | category    | sales_amount | quantity |
| ---: | :------------------ | ----------: | ---------: | :---------- | -----------: | -------: |
|    0 | 2023-01-01 00:00:00 |          52 |         45 | Food        |       490.76 |        7 |
|    1 | 2023-01-02 00:00:00 |          93 |         41 | Electronics |       453.94 |        5 |
|    2 | 2023-01-03 00:00:00 |          15 |         29 | Home        |       994.51 |        5 |
|    3 | 2023-01-04 00:00:00 |          72 |         15 | Electronics |       184.17 |        7 |
|    4 | 2023-01-05 00:00:00 |          61 |         45 | Food        |        27.89 |        9 |

</div>

```python {.polars linenums="1" title="Check Product DataFrame"}
print(f"Product DataFrame: {df_product_pl.shape[0]}")
print(df_product_pl.head(5))
print(df_product_pl.head(5).to_pandas().to_markdown())
```

<div class="result" markdown>

```txt
Product DataFrame: 50
```

```txt
shape: (5, 5)
┌────────────┬──────────────┬────────┬──────────┬─────────────┐
│ product_id ┆ product_name ┆ price  ┆ category ┆ supplier_id │
│ ---        ┆ ---          ┆ ---    ┆ ---      ┆ ---         │
│ i64        ┆ str          ┆ f64    ┆ str      ┆ i64         │
╞════════════╪══════════════╪════════╪══════════╪═════════════╡
│ 1          ┆ Product 1    ┆ 257.57 ┆ Food     ┆ 8           │
│ 2          ┆ Product 2    ┆ 414.96 ┆ Clothing ┆ 5           │
│ 3          ┆ Product 3    ┆ 166.82 ┆ Clothing ┆ 8           │
│ 4          ┆ Product 4    ┆ 448.81 ┆ Food     ┆ 4           │
│ 5          ┆ Product 5    ┆ 200.71 ┆ Food     ┆ 8           │
└────────────┴──────────────┴────────┴──────────┴─────────────┘
```

|      | product_id | product_name |  price | category | supplier_id |
| ---: | ---------: | :----------- | -----: | :------- | ----------: |
|    0 |          1 | Product 1    | 257.57 | Food     |           8 |
|    1 |          2 | Product 2    | 414.96 | Clothing |           5 |
|    2 |          3 | Product 3    | 166.82 | Clothing |           8 |
|    3 |          4 | Product 4    | 448.81 | Food     |           4 |
|    4 |          5 | Product 5    | 200.71 | Food     |           8 |

</div>

```python {.polars linenums="1" title="Check Customer DataFrame"}
print(f"Customer DataFrame: {df_customer_pl.shape[0]}")
print(df_customer_pl.head(5))
print(df_customer_pl.head(5).to_pandas().to_markdown())
```

<div class="result" markdown>

```txt
Customer DataFrame: 100
```

```txt
shape: (5, 5)
┌─────────────┬───────────────┬─────────────┬───────┬─────────────┐
│ customer_id ┆ customer_name ┆ city        ┆ state ┆ segment     │
│ ---         ┆ ---           ┆ ---         ┆ ---   ┆ ---         │
│ i64         ┆ str           ┆ str         ┆ str   ┆ str         │
╞═════════════╪═══════════════╪═════════════╪═══════╪═════════════╡
│ 1           ┆ Customer 1    ┆ Phoenix     ┆ NY    ┆ Corporate   │
│ 2           ┆ Customer 2    ┆ Phoenix     ┆ CA    ┆ Home Office │
│ 3           ┆ Customer 3    ┆ Phoenix     ┆ NY    ┆ Home Office │
│ 4           ┆ Customer 4    ┆ Los Angeles ┆ NY    ┆ Consumer    │
│ 5           ┆ Customer 5    ┆ Los Angeles ┆ IL    ┆ Home Office │
└─────────────┴───────────────┴─────────────┴───────┴─────────────┘
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

### Polars

In Polars, we can use the [`.filter()`][polars-filter] method to filter rows based on specific conditions. The syntax is similar to Pandas, where we can specify the condition using column expressions. In the below example, we filter for sales in the "Electronics" category.

For more information about filtering in Polars, see the [Polars documentation on filtering][polars-filtering].

```python {.polars linenums="1" title="Filter sales for a specific category"}
electronics_sales_pl: pl.DataFrame = df_sales_pl.filter(df_sales_pl["category"] == "Electronics")
print(f"Number of Electronics Sales: {len(electronics_sales_pl)}")
print(electronics_sales_pl.head(5))
print(electronics_sales_pl.head(5).to_pandas().to_markdown())
```

<div class="result" markdown>

```txt
Number of Electronics Sales: 28
```

```txt
shape: (5, 6)
┌─────────────────────┬─────────────┬────────────┬─────────────┬──────────────┬──────────┐
│ date                ┆ customer_id ┆ product_id ┆ category    ┆ sales_amount ┆ quantity │
│ ---                 ┆ ---         ┆ ---        ┆ ---         ┆ ---          ┆ ---      │
│ datetime[ns]        ┆ i64         ┆ i64        ┆ str         ┆ f64          ┆ i64      │
╞═════════════════════╪═════════════╪════════════╪═════════════╪══════════════╪══════════╡
│ 2023-01-02 00:00:00 ┆ 93          ┆ 41         ┆ Electronics ┆ 453.94       ┆ 5        │
│ 2023-01-04 00:00:00 ┆ 72          ┆ 15         ┆ Electronics ┆ 184.17       ┆ 7        │
│ 2023-01-09 00:00:00 ┆ 75          ┆ 9          ┆ Electronics ┆ 746.73       ┆ 2        │
│ 2023-01-11 00:00:00 ┆ 88          ┆ 1          ┆ Electronics ┆ 314.98       ┆ 9        │
│ 2023-01-12 00:00:00 ┆ 24          ┆ 44         ┆ Electronics ┆ 547.11       ┆ 8        │
└─────────────────────┴─────────────┴────────────┴─────────────┴──────────────┴──────────┘
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

### Polars

When it comes to numerical filtering in Polars, the process is similar to the previous example, where we use the [`.filter()`][polars-filter] method to filter rows based on a given condition, but here we use a numerical value instead of a string value. In the below example, we filter for sales amounts greater than `500`.

Also note here that we have used the [`pl.col()`][polars-col] function to specify the column we want to filter on. This is different from the previous examples, where we used the column name directly. The use of [`pl.col()`][polars-col] is a common practice in Polars to specify the column name in a more readable way.

```python {.polars linenums="1" title="Filter for high value transactions"}
high_value_sales_pl: pl.DataFrame = df_sales_pl.filter(pl.col("sales_amount") > 500)
print(f"Number of high-value Sales: {len(high_value_sales_pl)}")
print(high_value_sales_pl.head(5))
print(high_value_sales_pl.head(5).to_pandas().to_markdown())
```

<div class="result" markdown>

```txt
Number of high-value Sales: 43
```

```txt
shape: (5, 6)
┌─────────────────────┬─────────────┬────────────┬─────────────┬──────────────┬──────────┐
│ date                ┆ customer_id ┆ product_id ┆ category    ┆ sales_amount ┆ quantity │
│ ---                 ┆ ---         ┆ ---        ┆ ---         ┆ ---          ┆ ---      │
│ datetime[ns]        ┆ i64         ┆ i64        ┆ str         ┆ f64          ┆ i64      │
╞═════════════════════╪═════════════╪════════════╪═════════════╪══════════════╪══════════╡
│ 2023-01-03 00:00:00 ┆ 15          ┆ 29         ┆ Home        ┆ 994.51       ┆ 5        │
│ 2023-01-09 00:00:00 ┆ 75          ┆ 9          ┆ Electronics ┆ 746.73       ┆ 2        │
│ 2023-01-10 00:00:00 ┆ 75          ┆ 24         ┆ Books       ┆ 723.73       ┆ 6        │
│ 2023-01-12 00:00:00 ┆ 24          ┆ 44         ┆ Electronics ┆ 547.11       ┆ 8        │
│ 2023-01-13 00:00:00 ┆ 3           ┆ 8          ┆ Clothing    ┆ 513.73       ┆ 5        │
└─────────────────────┴─────────────┴────────────┴─────────────┴──────────────┴──────────┘
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


### Polars

To select specific columns in Polars, we can use the [`.select()`][polars-select] method to specify the columns we want to keep in the DataFrame. This allows us to create a new DataFrame with only the relevant columns.

```python {.polars linenums="1" title="Select specific columns"}
sales_summary_pl: pl.DataFrame = df_sales_pl.select(["date", "category", "sales_amount"])
print(f"Sales Summary DataFrame: {len(sales_summary_pl)}")
print(sales_summary_pl.head(5))
print(sales_summary_pl.head(5).to_pandas().to_markdown())
```

<div class="result" markdown>

```txt
Sales Summary DataFrame: 100
```

```txt
shape: (5, 3)
┌─────────────────────┬─────────────┬──────────────┐
│ date                ┆ category    ┆ sales_amount │
│ ---                 ┆ ---         ┆ ---          │
│ datetime[ns]        ┆ str         ┆ f64          │
╞═════════════════════╪═════════════╪══════════════╡
│ 2023-01-01 00:00:00 ┆ Food        ┆ 490.76       │
│ 2023-01-02 00:00:00 ┆ Electronics ┆ 453.94       │
│ 2023-01-03 00:00:00 ┆ Home        ┆ 994.51       │
│ 2023-01-04 00:00:00 ┆ Electronics ┆ 184.17       │
│ 2023-01-05 00:00:00 ┆ Food        ┆ 27.89        │
└─────────────────────┴─────────────┴──────────────┘
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

### Polars

In Polars, we can use the [`.select()`][polars-select] method to perform aggregation operations on DataFrames. This method allows us to apply multiple aggregation functions to different columns in a single operation.

```python {.polars linenums="1" title="Basic aggregation"}
sales_stats_pl: pl.DataFrame = df_sales_pl.select(
    pl.col("sales_amount").sum().alias("sales_sum"),
    pl.col("sales_amount").mean().alias("sales_mean"),
    pl.col("sales_amount").min().alias("sales_min"),
    pl.col("sales_amount").max().alias("sales_max"),
    pl.col("quantity").sum().alias("quantity_sum"),
    pl.col("quantity").mean().alias("quantity_mean"),
    pl.col("quantity").min().alias("quantity_min"),
    pl.col("quantity").max().alias("quantity_max"),
)
print(f"Sales Statistics: {len(sales_stats_pl)}")
print(sales_stats_pl)
print(sales_stats_pl.to_pandas().to_markdown())
```

<div class="result" markdown>

```txt
Sales Statistics: 1
```

```txt
shape: (1, 8)
┌───────────┬────────────┬───────────┬───────────┬──────────────┬───────────────┬──────────────┬──────────────┐
│ sales_sum ┆ sales_mean ┆ sales_min ┆ sales_max ┆ quantity_sum ┆ quantity_mean ┆ quantity_min ┆ quantity_max │
│ ---       ┆ ---        ┆ ---       ┆ ---       ┆ ---          ┆ ---           ┆ ---          ┆ ---          │
│ f64       ┆ f64        ┆ f64       ┆ f64       ┆ i64          ┆ f64           ┆ i64          ┆ i64          │
╞═══════════╪════════════╪═══════════╪═══════════╪══════════════╪═══════════════╪══════════════╪══════════════╡
│ 48227.05  ┆ 482.2705   ┆ 15.13     ┆ 994.61    ┆ 464          ┆ 4.64          ┆ 1            ┆ 9            │
└───────────┴────────────┴───────────┴───────────┴──────────────┴───────────────┴──────────────┴──────────────┘
```

|      | sales_sum | sales_mean | sales_min | sales_max | quantity_sum | quantity_mean | quantity_min | quantity_max |
| ---: | --------: | ---------: | --------: | --------: | -----------: | ------------: | -----------: | -----------: |
|    0 |     48227 |     482.27 |     15.13 |    994.61 |          464 |          4.64 |            1 |            9 |

</div>

It is also possible to group the data by a specific column and then apply aggregation functions to summarize the data by group.

### Polars

In Polars, we can use the [`.group_by()`][polars-groupby] method to group data by one or more columns and then apply aggregation functions using the [`.agg()`][polars-groupby-agg] method.

```python {.polars linenums="1" title="Group by category and aggregate"}
category_sales_pl: pl.DataFrame = df_sales_pl.group_by("category").agg(
    pl.col("sales_amount").sum().alias("total_sales"),
    pl.col("sales_amount").mean().alias("average_sales"),
    pl.col("sales_amount").count().alias("transaction_count"),
    pl.col("quantity").sum().alias("total_quantity"),
)
print(f"Category Sales Summary: {len(category_sales_pl)}")
print(category_sales_pl.head(5))
print(category_sales_pl.head(5).to_pandas().to_markdown())
```

<div class="result" markdown>

```txt
Category Sales Summary: 5
```

```txt
shape: (5, 5)
┌─────────────┬─────────────┬───────────────┬───────────────────┬────────────────┐
│ category    ┆ total_sales ┆ average_sales ┆ transaction_count ┆ total_quantity │
│ ---         ┆ ---         ┆ ---           ┆ ---               ┆ ---            │
│ str         ┆ f64         ┆ f64           ┆ u32               ┆ i64            │
╞═════════════╪═════════════╪═══════════════╪═══════════════════╪════════════════╡
│ Food        ┆ 12995.57    ┆ 541.482083    ┆ 24                ┆ 115            │
│ Electronics ┆ 11407.45    ┆ 407.408929    ┆ 28                ┆ 147            │
│ Books       ┆ 10154.83    ┆ 441.514348    ┆ 23                ┆ 100            │
│ Home        ┆ 6343.89     ┆ 704.876667    ┆ 9                 ┆ 40             │
│ Clothing    ┆ 7325.31     ┆ 457.831875    ┆ 16                ┆ 62             │
└─────────────┴─────────────┴───────────────┴───────────────────┴────────────────┘
```

|      | category    | total_sales | average_sales | transaction_count | total_quantity |
| ---: | :---------- | ----------: | ------------: | ----------------: | -------------: |
|    0 | Food        |     12995.6 |       541.482 |                24 |            115 |
|    1 | Electronics |     11407.5 |       407.409 |                28 |            147 |
|    2 | Books       |     10154.8 |       441.514 |                23 |            100 |
|    3 | Home        |     6343.89 |       704.877 |                 9 |             40 |
|    4 | Clothing    |     7325.31 |       457.832 |                16 |             62 |

</div>

We can rename the columns for clarity by simply assigning new names.

### Polars

In Polars, we can use the [`.rename()`][polars-rename] method to rename columns in a DataFrame. This allows us to provide more descriptive names for the aggregated columns.

```python {.polars linenums="1" title="Rename columns for clarity"}
category_sales_renamed_pl: pl.DataFrame = category_sales_pl.rename(
    {
        "total_sales": "Total Sales",
        "average_sales": "Average Sales",
        "transaction_count": "Transaction Count",
        "total_quantity": "Total Quantity",
    }
)
print(f"Renamed Category Sales Summary: {len(category_sales_renamed_pl)}")
print(category_sales_renamed_pl.head(5))
print(category_sales_renamed_pl.head(5).to_pandas().to_markdown())
```

<div class="result" markdown>

```txt
Renamed Category Sales Summary: 5
```

```txt
shape: (5, 5)
┌─────────────┬─────────────┬───────────────┬───────────────────┬────────────────┐
│ category    ┆ Total Sales ┆ Average Sales ┆ Transaction Count ┆ Total Quantity │
│ ---         ┆ ---         ┆ ---           ┆ ---               ┆ ---            │
│ str         ┆ f64         ┆ f64           ┆ u32               ┆ i64            │
╞═════════════╪═════════════╪═══════════════╪═══════════════════╪════════════════╡
│ Food        ┆ 12995.57    ┆ 541.482083    ┆ 24                ┆ 115            │
│ Electronics ┆ 11407.45    ┆ 407.408929    ┆ 28                ┆ 147            │
│ Books       ┆ 10154.83    ┆ 441.514348    ┆ 23                ┆ 100            │
│ Home        ┆ 6343.89     ┆ 704.876667    ┆ 9                 ┆ 40             │
│ Clothing    ┆ 7325.31     ┆ 457.831875    ┆ 16                ┆ 62             │
└─────────────┴─────────────┴───────────────┴───────────────────┴────────────────┘
```

|      | category    | Total Sales | Average Sales | Transaction Count | Total Quantity |
| ---: | :---------- | ----------: | ------------: | ----------------: | -------------: |
|    0 | Food        |     12995.6 |       541.482 |                24 |            115 |
|    1 | Electronics |     11407.5 |       407.409 |                28 |            147 |
|    2 | Books       |     10154.8 |       441.514 |                23 |            100 |
|    3 | Home        |     6343.89 |       704.877 |                 9 |             40 |
|    4 | Clothing    |     7325.31 |       457.832 |                16 |             62 |

</div>

Having aggregated the data, we can now visualize the results using [Plotly][plotly]. This allows us to create interactive visualizations that can help us better understand the data. The simplest way to do this is to use the [Plotly Express][plotly-express] module, which provides a high-level interface for creating visualizations. Here, we have utilised the [`px.bar()`][plotly-bar] function to create a bar chart of the total sales by category.

### Polars

Plotly is also able to receive a Polars DataFrame, so we can use the results of the aggregation directly.

```python {.polars linenums="1" title="Plot the results"}
fig: go.Figure = px.bar(
    data_frame=category_sales_renamed_pl,
    x="category",
    y="Total Sales",
    title="Total Sales by Category",
    text="Transaction Count",
    labels={"Total Sales": "Total Sales ($)", "category": "Product Category"},
)
fig.write_html("images/pt2_total_sales_by_category_pl.html", include_plotlyjs="cdn", full_html=True)
fig.show()
```

<div class="result" markdown>


</div>



## 3. Joining

The third section will demonstrate how to join DataFrames to combine data from different sources. This is a common operation in data analysis, allowing us to enrich our data with additional information.

Here, we will join the `sales` DataFrame with the `product` DataFrame to get additional information about the products sold.

### Polars

In Polars, we can use the [`.join()`][polars-join] method to combine rows from two or more DataFrames based on a related column between them. In this case, we will join the `sales` DataFrame with the `product` DataFrame on the `product_id` column.

```python {.polars linenums="1" title="Join sales with product data"}
sales_with_product_pl: pl.DataFrame = df_sales_pl.join(
    df_product_pl.select(["product_id", "product_name", "price"]),
    on="product_id",
    how="left",
)
print(f"Sales with Product Information: {len(sales_with_product_pl)}")
print(sales_with_product_pl.head(5))
print(sales_with_product_pl.head(5).to_pandas().to_markdown())
```

<div class="result" markdown>

```txt
Sales with Product Information: 100
```

```txt
shape: (5, 8)
┌─────────────────────┬─────────────┬────────────┬─────────────┬──────────────┬──────────┬──────────────┬────────┐
│ date                ┆ customer_id ┆ product_id ┆ category    ┆ sales_amount ┆ quantity ┆ product_name ┆ price  │
│ ---                 ┆ ---         ┆ ---        ┆ ---         ┆ ---          ┆ ---      ┆ ---          ┆ ---    │
│ datetime[ns]        ┆ i64         ┆ i64        ┆ str         ┆ f64          ┆ i64      ┆ str          ┆ f64    │
╞═════════════════════╪═════════════╪════════════╪═════════════╪══════════════╪══════════╪══════════════╪════════╡
│ 2023-01-01 00:00:00 ┆ 52          ┆ 45         ┆ Food        ┆ 490.76       ┆ 7        ┆ Product 45   ┆ 493.14 │
│ 2023-01-02 00:00:00 ┆ 93          ┆ 41         ┆ Electronics ┆ 453.94       ┆ 5        ┆ Product 41   ┆ 193.39 │
│ 2023-01-03 00:00:00 ┆ 15          ┆ 29         ┆ Home        ┆ 994.51       ┆ 5        ┆ Product 29   ┆ 80.07  │
│ 2023-01-04 00:00:00 ┆ 72          ┆ 15         ┆ Electronics ┆ 184.17       ┆ 7        ┆ Product 15   ┆ 153.67 │
│ 2023-01-05 00:00:00 ┆ 61          ┆ 45         ┆ Food        ┆ 27.89        ┆ 9        ┆ Product 45   ┆ 493.14 │
└─────────────────────┴─────────────┴────────────┴─────────────┴──────────────┴──────────┴──────────────┴────────┘
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

### Polars

This process is similar to the previous step, but now we will extend the `sales_with_product` DataFrame to join it with the `customer` DataFrame on the `customer_id` column. This will give us a complete view of the sales data, including product and customer details.

```python {.polars linenums="1" title="Join with customer information to get a complete view"}
complete_sales_pl: pl.DataFrame = sales_with_product_pl.join(
    df_customer_pl.select(["customer_id", "customer_name", "city", "state"]),
    on="customer_id",
    how="left",
)
print(f"Complete Sales Data with Customer Information: {len(complete_sales_pl)}")
print(complete_sales_pl.head(5))
print(complete_sales_pl.head(5).to_pandas().to_markdown())
```

<div class="result" markdown>

```txt
Complete Sales Data with Customer Information: 100
```

```txt
shape: (5, 11)
┌─────────────────────┬─────────────┬────────────┬─────────────┬──────────────┬──────────┬──────────────┬────────┬───────────────┬──────────┬───────┐
│ date                ┆ customer_id ┆ product_id ┆ category    ┆ sales_amount ┆ quantity ┆ product_name ┆ price  ┆ customer_name ┆ city     ┆ state │
│ ---                 ┆ ---         ┆ ---        ┆ ---         ┆ ---          ┆ ---      ┆ ---          ┆ ---    ┆ ---           ┆ ---      ┆ ---   │
│ datetime[ns]        ┆ i64         ┆ i64        ┆ str         ┆ f64          ┆ i64      ┆ str          ┆ f64    ┆ str           ┆ str      ┆ str   │
╞═════════════════════╪═════════════╪════════════╪═════════════╪══════════════╪══════════╪══════════════╪════════╪═══════════════╪══════════╪═══════╡
│ 2023-01-01 00:00:00 ┆ 52          ┆ 45         ┆ Food        ┆ 490.76       ┆ 7        ┆ Product 45   ┆ 493.14 ┆ Customer 52   ┆ Phoenis  ┆ TX    │
│ 2023-01-02 00:00:00 ┆ 93          ┆ 41         ┆ Electronics ┆ 453.94       ┆ 5        ┆ Product 41   ┆ 193.39 ┆ Customer 93   ┆ New York ┆ TX    │
│ 2023-01-03 00:00:00 ┆ 15          ┆ 29         ┆ Home        ┆ 994.51       ┆ 5        ┆ Product 29   ┆ 80.07  ┆ Customer 15   ┆ New York ┆ CA    │
│ 2023-01-04 00:00:00 ┆ 72          ┆ 15         ┆ Electronics ┆ 184.17       ┆ 7        ┆ Product 15   ┆ 153.67 ┆ Customer 72   ┆ Houston  ┆ IL    │
│ 2023-01-05 00:00:00 ┆ 61          ┆ 45         ┆ Food        ┆ 27.89        ┆ 9        ┆ Product 45   ┆ 493.14 ┆ Customer 61   ┆ Phoenix  ┆ IL    │
└─────────────────────┴─────────────┴────────────┴─────────────┴──────────────┴──────────┴──────────────┴────────┴───────────────┴──────────┴───────┘
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

### Polars

In Polars, we can calculate the revenue for each sale by multiplying the `price` and `quantity` columns. We can then compare this calculated revenue with the `sales_amount` column to identify any discrepancies.

Notice here that the syntax for Polars uses the [`.with_columns`][polars-with-columns] method to add new multiple columns to the DataFrame simultaneously. This method takes a list of expressions, where each expression defines a new column.

```python {.polars linenums="1" title="Calculate revenue and compare with sales amount"}
complete_sales_pl: pl.DataFrame = complete_sales_pl.with_columns(
    (pl.col("price") * pl.col("quantity")).alias("calculated_revenue"),
    (pl.col("sales_amount") - (pl.col("price") * pl.col("quantity"))).alias("price_difference"),
)
print(f"Complete Sales Data with Calculated Revenue and Price Difference: {len(complete_sales_pl)}")
print(complete_sales_pl.select(["sales_amount", "price", "quantity", "calculated_revenue", "price_difference"]).head(5))
print(
    complete_sales_pl.select(["sales_amount", "price", "quantity", "calculated_revenue", "price_difference"])
    .head(5)
    .to_pandas()
    .to_markdown()
)
```

<div class="result" markdown>

```txt
Complete Sales Data with Calculated Revenue and Price Difference: 100
```

```txt
┌──────────────┬────────┬──────────┬────────────────────┬──────────────────┐
│ sales_amount ┆ price  ┆ quantity ┆ calculated_revenue ┆ price_difference │
│ ---          ┆ ---    ┆ ---      ┆ ---                ┆ ---              │
│ f64          ┆ f64    ┆ i64      ┆ f64                ┆ f64              │
╞══════════════╪════════╪══════════╪════════════════════╪══════════════════╡
│ 490.76       ┆ 493.14 ┆ 7        ┆ 3451.98            ┆ -2961.22         │
│ 453.94       ┆ 193.39 ┆ 5        ┆ 966.95             ┆ -513.01          │
│ 994.51       ┆ 80.07  ┆ 5        ┆ 400.35             ┆ 594.16           │
│ 184.17       ┆ 153.67 ┆ 7        ┆ 1075.69            ┆ -891.52          │
│ 27.89        ┆ 493.14 ┆ 9        ┆ 4438.26            ┆ -4410.37         │
└──────────────┴────────┴──────────┴────────────────────┴──────────────────┘
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

### Polars

In Polars, we can use the [`.group_by()`][polars-groupby] method to group the data by the `date` column, followed by the [`.agg()`][polars-groupby-agg] method to calculate the total sales for each day. This will then set us up for further time-based calculations in the following steps.

```python {.polars linenums="1" title="Time-based window function"}
df_sales_pl: pl.DataFrame = df_sales_pl.with_columns(pl.col("date").cast(pl.Date))
daily_sales_pl: pl.DataFrame = (
    df_sales_pl.group_by("date").agg(pl.col("sales_amount").sum().alias("total_sales")).sort("date")
)
print(f"Daily Sales Summary: {len(daily_sales_pl)}")
print(daily_sales_pl.head(5))
print(daily_sales_pl.head(5).to_pandas().to_markdown())
```

<div class="result" markdown>

```txt
Daily Sales Summary: 100
```

```txt
shape: (5, 2)
┌────────────┬─────────────┐
│ date       ┆ total_sales │
│ ---        ┆ ---         │
│ date       ┆ f64         │
╞════════════╪═════════════╡
│ 2023-01-01 ┆ 490.76      │
│ 2023-01-02 ┆ 453.94      │
│ 2023-01-03 ┆ 994.51      │
│ 2023-01-04 ┆ 184.17      │
│ 2023-01-05 ┆ 27.89       │
└────────────┴─────────────┘
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

### Polars

In Polars, we can use the [`.shift()`][polars-shift] method to calculate the lag and lead values for the sales amount. This method shifts the values in a column by a specified number of periods, allowing us to create lag and lead columns.

Note that the [`.shift()`][polars-shift] method simply shifts the values in the column by a number of rows up or down, so we can use it to create lag and lead columns. This function itself does not need to be ordered because it assumes that the DataFrame is already ordered. However, if you want it to be ordered, you can use the [`.sort()`][polars-sort] method before applying [`.shift()`][polars-shift].

```python {.polars linenums="1" title="Calculate lag and lead"}
daily_sales_pl: pl.DataFrame = daily_sales_pl.with_columns(
    pl.col("total_sales").shift(1).alias("previous_day_sales"),
    pl.col("total_sales").shift(-1).alias("next_day_sales"),
)
print(f"Daily Sales with Lag and Lead: {len(daily_sales_pl)}")
print(daily_sales_pl.head(5))
print(daily_sales_pl.head(5).to_pandas().to_markdown())
```

<div class="result" markdown>

```txt
Daily Sales with Lag and Lead: 100
```

```txt
shape: (5, 4)
┌────────────┬─────────────┬────────────────────┬────────────────┐
│ date       ┆ total_sales ┆ previous_day_sales ┆ next_day_sales │
│ ---        ┆ ---         ┆ ---                ┆ ---            │
│ date       ┆ f64         ┆ f64                ┆ f64            │
╞════════════╪═════════════╪════════════════════╪════════════════╡
│ 2023-01-01 ┆ 490.76      ┆ null               ┆ 453.94         │
│ 2023-01-02 ┆ 453.94      ┆ 490.76             ┆ 994.51         │
│ 2023-01-03 ┆ 994.51      ┆ 453.94             ┆ 184.17         │
│ 2023-01-04 ┆ 184.17      ┆ 994.51             ┆ 27.89          │
│ 2023-01-05 ┆ 27.89       ┆ 184.17             ┆ 498.95         │
└────────────┴─────────────┴────────────────────┴────────────────┘
```

|      | date                | total_sales | previous_day_sales | next_day_sales |
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

### Polars

In Polars, we can calculate the day-over-day change in sales by subtracting the `previous_day_sales` column from the `total_sales` column. We can also calculate the percentage change in sales using the formula:

```txt
((current_day_sales - previous_day_sales) / previous_day_sales) * 100
```

```python {.polars linenums="1" title="Calculate day-over-day change"}
daily_sales_pl: pl.DataFrame = daily_sales_pl.with_columns(
    (pl.col("total_sales") - pl.col("previous_day_sales")).alias("day_over_day_change"),
    (pl.col("total_sales") / pl.col("previous_day_sales") - 1).alias("pct_change") * 100,
)
print(f"Daily Sales with Day-over-Day Change: {len(daily_sales_pl)}")
print(daily_sales_pl.head(5))
print(daily_sales_pl.head(5).to_pandas().to_markdown())
```

<div class="result" markdown>

```txt
Daily Sales with Day-over-Day Change: 100
```

```txt
shape: (5, 6)
┌────────────┬─────────────┬────────────────────┬────────────────┬─────────────────────┬────────────┐
│ date       ┆ total_sales ┆ previous_day_sales ┆ next_day_sales ┆ day_over_day_change ┆ pct_change │
│ ---        ┆ ---         ┆ ---                ┆ ---            ┆ ---                 ┆ ---        │
│ date       ┆ f64         ┆ f64                ┆ f64            ┆ f64                 ┆ f64        │
╞════════════╪═════════════╪════════════════════╪════════════════╪═════════════════════╪════════════╡
│ 2023-01-01 ┆ 490.76      ┆ null               ┆ 453.94         ┆ null                ┆ null       │
│ 2023-01-02 ┆ 453.94      ┆ 490.76             ┆ 994.51         ┆ -36.82              ┆ -7.502649  │
│ 2023-01-03 ┆ 994.51      ┆ 453.94             ┆ 184.17         ┆ 540.57              ┆ 119.08402  │
│ 2023-01-04 ┆ 184.17      ┆ 994.51             ┆ 27.89          ┆ -810.34             ┆ -81.481333 │
│ 2023-01-05 ┆ 27.89       ┆ 184.17             ┆ 498.95         ┆ -156.28             ┆ -84.856383 │
└────────────┴─────────────┴────────────────────┴────────────────┴─────────────────────┴────────────┘
```

|      | date                | total_sales | previous_day_sales | next_day_sales | day_over_day_change | pct_change |
| ---: | :------------------ | ----------: | -----------------: | -------------: | ------------------: | ---------: |
|    0 | 2023-01-01 00:00:00 |      490.76 |                nan |         453.94 |                 nan |        nan |
|    1 | 2023-01-02 00:00:00 |      453.94 |             490.76 |         994.51 |              -36.82 |   -7.50265 |
|    2 | 2023-01-03 00:00:00 |      994.51 |             453.94 |         184.17 |              540.57 |    119.084 |
|    3 | 2023-01-04 00:00:00 |      184.17 |             994.51 |          27.89 |             -810.34 |   -81.4813 |
|    4 | 2023-01-05 00:00:00 |       27.89 |             184.17 |         498.95 |             -156.28 |   -84.8564 |

</div>

Next, we will calculate the rolling average of sales over a 7-day window. Rolling averages (aka moving averages) are useful for smoothing out short-term fluctuations and highlighting longer-term trends in the data. This is particularly useful in time series analysis, where we want to understand the underlying trend in the data without being overly influenced by short-term variations. It is also a very common technique used in financial analysis to analyze stock prices, sales data, and other time series data.

### Polars

In Polars, we can calculate the 7-day moving average of sales using the [`.rolling_mean()`][polars-rolling-mean] method. This method allows us to specify a window size (in this case, `window_size=7` which is 7 days) and calculate the mean over that window. The `min_samples=1` parameter ensures that we get a value even if there are fewer than 7 days of data available at the start of the series.

```python {.polars linenums="1" title="Calculate 7-day moving average"}
daily_sales_pl: pl.DataFrame = daily_sales_pl.with_columns(
    pl.col("total_sales").rolling_mean(window_size=7, min_samples=1).alias("7d_moving_avg"),
)
print(f"Daily Sales with 7-Day Moving Average: {len(daily_sales_pl)}")
print(daily_sales_pl.head(5))
print(daily_sales_pl.head(5).to_pandas().to_markdown())
```

<div class="result" markdown>

```txt
Daily Sales with 7-Day Moving Average: 100
```

```txt
shape: (5, 7)
┌────────────┬─────────────┬────────────────────┬────────────────┬─────────────────────┬────────────┬───────────────┐
│ date       ┆ total_sales ┆ previous_day_sales ┆ next_day_sales ┆ day_over_day_change ┆ pct_change ┆ 7d_moving_avg │
│ ---        ┆ ---         ┆ ---                ┆ ---            ┆ ---                 ┆ ---        ┆ ---           │
│ date       ┆ f64         ┆ f64                ┆ f64            ┆ f64                 ┆ f64        ┆ f64           │
╞════════════╪═════════════╪════════════════════╪════════════════╪═════════════════════╪════════════╪═══════════════╡
│ 2023-01-01 ┆ 490.76      ┆ null               ┆ 453.94         ┆ null                ┆ null       ┆ 490.76        │
│ 2023-01-02 ┆ 453.94      ┆ 490.76             ┆ 994.51         ┆ -36.82              ┆ -7.502649  ┆ 472.35        │
│ 2023-01-03 ┆ 994.51      ┆ 453.94             ┆ 184.17         ┆ 540.57              ┆ 119.08402  ┆ 646.403333    │
│ 2023-01-04 ┆ 184.17      ┆ 994.51             ┆ 27.89          ┆ -810.34             ┆ -81.481333 ┆ 530.845       │
│ 2023-01-05 ┆ 27.89       ┆ 184.17             ┆ 498.95         ┆ -156.28             ┆ -84.856383 ┆ 430.254       │
└────────────┴─────────────┴────────────────────┴────────────────┴─────────────────────┴────────────┴───────────────┘
```

|      | date                | total_sales | previous_day_sales | next_day_sales | day_over_day_change | pct_change | 7d_moving_avg |
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

### Polars

Plotly is easily able to handle Polars DataFrames, so we can directly parse the columns from the DataFrame to create the traces for the daily sales and the 7-day moving average.

```python {.polars linenums="1" title="Plot results"}
fig: go.Figure = (
    go.Figure()
    .add_trace(
        go.Scatter(
            x=daily_sales_pl["date"],
            y=daily_sales_pl["total_sales"],
            mode="lines",
            name="Daily Sales",
        )
    )
    .add_trace(
        go.Scatter(
            x=daily_sales_pl["date"],
            y=daily_sales_pl["7d_moving_avg"],
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
fig.write_html("images/pt4_daily_sales_with_7d_avg_pl.html", include_plotlyjs="cdn", full_html=True)
fig.show()
```

<div class="result" markdown>


</div>



## 5. Ranking and Partitioning

The fifth section will demonstrate how to rank and partition data. This is useful for identifying top performers, such as the highest spending customers or the most popular products.

### Polars

In Polars, we can use the [`.rank()`][polars-rank] method to rank values in a DataFrame. This method allows us to specify the ranking method (e.g., dense, average, min, max) and whether to rank in ascending or descending order.

```python {.polars linenums="1" title="Rank customers by total spending"}
customer_spending_pl: pl.DataFrame = (
    df_sales_pl.group_by("customer_id")
    .agg(pl.col("sales_amount").sum().alias("total_spending"))
    .with_columns(
        pl.col("total_spending").rank(method="dense", descending=True).alias("rank"),
    )
    .sort("rank")
)
print(f"Customer Spending Summary: {len(customer_spending_pl)}")
print(customer_spending_pl.head(5))
print(customer_spending_pl.head(5).to_pandas().to_markdown())
```

<div class="result" markdown>

```txt
Customer Spending Summary: 61
```

```txt
shape: (5, 3)
┌─────────────┬────────────────┬──────┐
│ customer_id ┆ total_spending ┆ rank │
│ ---         ┆ ---            ┆ ---  │
│ i64         ┆ f64            ┆ u32  │
╞═════════════╪════════════════╪══════╡
│ 15          ┆ 2297.55        ┆ 1    │
│ 4           ┆ 2237.49        ┆ 2    │
│ 62          ┆ 2177.35        ┆ 3    │
│ 60          ┆ 2086.09        ┆ 4    │
│ 21          ┆ 2016.95        ┆ 5    │
└─────────────┴────────────────┴──────┘
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

### Polars

In Polars, we can use the [`.rank()`][polars-rank] method to rank products within each category based on the total quantity sold. We first group the sales data by `category` and `product_id`, then aggregate the data for the [`.sum()`][polars-groupby-sum] of the `quantity` to find the `total_quantity` sold for each product. After that, we can use the [`.rank()`][polars-rank] method to rank the products within each category based on the total quantity sold. Finally, we can define the partitioning by using the [`.over()`][polars-over] method and parse'ing in `partition_by="category"`.

```python {.polars linenums="1" title="Rank products by quantity sold, by category"}
product_popularity_pl: pl.DataFrame = (
    df_sales_pl.group_by("category", "product_id")
    .agg(pl.sum("quantity").alias("total_quantity"))
    .with_columns(
        pl.col("total_quantity").rank(method="dense", descending=True).over(partition_by="category").alias("rank")
    )
    .sort("rank", "category")
)
print(f"Product Popularity Summary: {len(product_popularity_pl)}")
print(product_popularity_pl.head(10))
print(product_popularity_pl.head(10).to_pandas().to_markdown())
```

<div class="result" markdown>

```txt
Product Popularity Summary: 78
```

```txt
shape: (10, 4)
┌─────────────┬────────────┬────────────────┬──────┐
│ category    ┆ product_id ┆ total_quantity ┆ rank │
│ ---         ┆ ---        ┆ ---            ┆ ---  │
│ str         ┆ i64        ┆ i64            ┆ u32  │
╞═════════════╪════════════╪════════════════╪══════╡
│ Books       ┆ 11         ┆ 14             ┆ 1    │
│ Clothing    ┆ 7          ┆ 9              ┆ 1    │
│ Electronics ┆ 37         ┆ 16             ┆ 1    │
│ Food        ┆ 45         ┆ 34             ┆ 1    │
│ Home        ┆ 3          ┆ 10             ┆ 1    │
│ Books       ┆ 28         ┆ 9              ┆ 2    │
│ Clothing    ┆ 35         ┆ 8              ┆ 2    │
│ Electronics ┆ 35         ┆ 11             ┆ 2    │
│ Food        ┆ 1          ┆ 16             ┆ 2    │
│ Home        ┆ 48         ┆ 5              ┆ 2    │
└─────────────┴────────────┴────────────────┴──────┘
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
