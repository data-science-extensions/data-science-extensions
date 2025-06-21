# Data Querying for Data Scientists

### A Comprehensive Guide of using Pandas, SQL, PySpark, and Polars for Data Manipulation Techniques, with Practical Examples and Visualisations


## Introduction

Working as a Data Scientist or Data Engineer often involves querying data from various sources. There are many tools and libraries available to perform these tasks, each with its own strengths and weaknesses. Also, there are many different ways to achieve similar results, depending on the tool or library used. It's important to be familiar with these different methods to choose the best one for your specific use case.

This article provides a comprehensive guide on how to query data using different tools and libraries, including Pandas, SQL, PySpark, and Polars. Each section will cover the setup, data creation, and various querying techniques such as filtering, grouping, joining, window functions, ranking, and sorting. The output will be identical across all tools, but the transformations will be implemented using the specific syntax and features of each library. Therefore allowing you to compare the different approaches and understand the nuances of each method.


## The Different Libraries

### Pandas

[Pandas](https://pandas.pydata.org/) is a powerful data manipulation library in Python that provides data structures and functions for working with structured data. It is widely used for data analysis and manipulation tasks.

Historically, Pandas was one of the first libraries to provide a DataFrame structure, which is similar to a table in a relational database. It allows for easy data manipulation, filtering, grouping, and aggregation. Pandas is built on top of [NumPy](https://numpy.org/) and provides a high-level interface for working with data. It is particularly well-suited for small to medium-sized datasets and is often used in Data Science and Machine Learning workflows.

Pandas provides a rich set of functionalities for data manipulation, including filtering, grouping, joining, and window functions. It also integrates well with other libraries such as Matplotlib and Seaborn for data visualization, making it a popular choice among data scientists and analysts.


### SQL

### PySpark

### Polars


## Setup

Before we start querying data, we need to set up our environment. This includes importing the necessary libraries, creating sample data, and defining constants that will be used throughout the article. The following sections will guide you through this setup process. The code for this article is also available on GitHub: [querying-data](...).


=== "Pandas"

    ```python
    # StdLib Imports
    from typing import Any

    # Third Party Imports
    import numpy as np
    import pandas as pd
    from plotly import express as px, graph_objects as go

    # Set seed for reproducibility
    np.random.seed(42)

    # Determine the number of records to generate
    n_records = 100
    ```

=== "SQL"

    ```python
    # StdLib Imports
    import sqlite3
    from typing import Any

    # Third Party Imports
    import numpy as np
    import pandas as pd
    from plotly import express as px, graph_objects as go
    ```

=== "PySpark"

    ```python
    # StdLib Imports
    from typing import Any

    # Third Party Imports
    import numpy as np
    from plotly import express as px, graph_objects as go
    from pyspark.sql import (
        DataFrame as psDataFrame,
        SparkSession,
        Window,
        functions as F,
        types as T,
    )

    # Set seed for reproducibility
    np.random.seed(42)

    # Determine the number of records to generate
    n_records = 100
    ```

=== "Polars"

    ```python
    # StdLib Imports
    from typing import Any

    # Third Party Imports
    import numpy as np
    import polars as pl
    from plotly import express as px, graph_objects as go

    # Set seed for reproducibility
    np.random.seed(42)

    # Determine the number of records to generate
    n_records = 100
    ```

Once the setup is complete, we can proceed to create our sample data. This data will be used for querying and will be consistent across all libraries. All tables will be created from scratch with randomly generated data to simulate a real-world scenario. This is to ensure that the examples are self-contained and can be run without any external dependencies, and also there is no issues about data privacy or security.

For the below data creation steps, we will be defining the tables using Python dictionaries. Each dictionary will represent a table, with keys as column names and values as lists of data. We will then convert these dictionaries into DataFrames or equivalent structures in each library.

First, we will create a sales fact table. This table will contain information about sales transactions, including the date, customer ID, product ID, category, sales amount, and quantity sold.

```python
sales_data: dict[str, Any] = {
    "date": pd.date_range(start="2023-01-01", periods=n_records, freq="D"),
    "customer_id": np.random.randint(1, 100, n_records),
    "product_id": np.random.randint(1, 50, n_records),
    "category": np.random.choice(
        ["Electronics", "Clothing", "Food", "Books", "Home"], n_records
    ),
    "sales_amount": np.random.uniform(10, 1000, n_records).round(2),
    "quantity": np.random.randint(1, 10, n_records),
}
```

Next, we will create a product dimension table. This table will contain information about products, including the product ID, name, price, category, and supplier ID.

```python
# Create product dimension table
product_data: dict[str, Any] = {
    "product_id": np.arange(1, 51),
    "product_name": [f"Product {i}" for i in range(1, 51)],
    "price": np.random.uniform(10, 500, 50).round(2),
    "category": np.random.choice(
        ["Electronics", "Clothing", "Food", "Books", "Home"], 50
    ),
    "supplier_id": np.random.randint(1, 10, 50),
}
```

Finally, we will create a customer dimension table. This table will contain information about customers, including the customer ID, name, city, state, and segment.

```python
# Create customer dimension table
customer_data: dict[str, Any] = {
    "customer_id": np.arange(1, 101),
    "customer_name": [f"Customer {i}" for i in range(1, 101)],
    "city": np.random.choice(
        ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"], 100
    ),
    "state": np.random.choice(["NY", "CA", "IL", "TX", "AZ"], 100),
    "segment": np.random.choice(["Consumer", "Corporate", "Home Office"], 100),
}
```

Now that we have our sample data created, we can proceed to the querying section. Each of the following sections will demonstrate how to perform similar operations using the different libraries and methods, allowing you to compare and contrast their capabilities.


## Create the DataFrames

=== "Pandas"

    To create the dataframes in Pandas, we will use the data we generated earlier. We will parse the dictionaries into Pandas DataFrames, which will allow us to perform various data manipulation tasks.

    ```python
    df_sales_pd = pd.DataFrame(sales_data)
    df_product_pd = pd.DataFrame(product_data)
    df_customer_pd = pd.DataFrame(customer_data)
    ```

    Once the data is created, we can check that it has been loaded correctly by displaying the first few rows of each DataFrame.

    ```python
    print(f"Sales DataFrame: {len(df_sales_pd)}")
    display(df_sales_pd.head(10))
    ```

    ```python
    print(f"Product DataFrame: {len(df_product_pd)}")
    display(df_product_pd.head(10))
    ```

    ```python
    print(f"Customer DataFrame: {len(df_customer_pd)}")
    display(df_customer_pd.head(10))
    ```

=== "SQL"

    ```python
    # Creates SQLite database and tables
    conn: sqlite3.Connection = sqlite3.connect(":memory:")
    df_sales_pd.to_sql("sales", conn, index=False, if_exists="replace")
    df_product_pd.to_sql("product", conn, index=False, if_exists="replace")
    df_customer_pd.to_sql("customer", conn, index=False, if_exists="replace")
    ```

    ```python
    print("Sales Table:")
    display(pd.read_sql("SELECT * FROM sales LIMIT 5", conn))
    ```

    ```python
    print("Product Table:")
    display(pd.read_sql("SELECT * FROM product LIMIT 5", conn))
    ```

    ```python
    print("Customer Table:")
    display(pd.read_sql("SELECT * FROM customer LIMIT 5", conn))
    ```

=== "PySpark"

    ```python
    spark: SparkSession = SparkSession.builder.appName("SalesAnalysis").getOrCreate()
    ```

    ```python
    df_sales_ps: psDataFrame = spark.createDataFrame(df_sales_pd)
    df_product_ps: psDataFrame = spark.createDataFrame(df_product_pd)
    df_customer_ps: psDataFrame = spark.createDataFrame(df_customer_pd)
    ```

    ```python
    print(f"Sales DataFrame: {df_sales_ps.count()}")
    df_sales_ps.show(10)
    ```

    ```python
    print(f"Product DataFrame: {df_product_ps.count()}")
    df_product_ps.show(10)
    ```

    ```python
    print(f"Customer DataFrame: {df_customer_ps.count()}")
    df_customer_ps.show(10)
    ```

=== "Polars"

    ```python
    df_sales_pl: pl.DataFrame = pl.DataFrame(sales_data)
    df_product_pl: pl.DataFrame = pl.DataFrame(product_data)
    df_customer_pl: pl.DataFrame = pl.DataFrame(customer_data)
    ```

    ```python
    print(f"Sales DataFrame: {df_sales_pl.shape[0]}")
    display(df_sales_pl.head(10))
    ```

    ```python
    print(f"Product DataFrame: {df_product_pl.shape[0]}")
    display(df_product_pl.head(10))
    ```

    ```python
    print(f"Customer DataFrame: {df_customer_pl.shape[0]}")
    display(df_customer_pl.head(10))
    ```


## 1. Filtering and Selecting

The first section will demonstrate how to filter and select data from the DataFrames. This is a common operation in data analysis, allowing us to focus on specific subsets of the data.

=== "Pandas"

    In Pandas, we can use boolean indexing to filter rows based on specific conditions. As you can see in this first example, this looks like using square brackets, within which we define a column and a condition. In the below example, we can use string values to filter categorical data.

    ```python
    # Filter sales data for specific category
    electronics_sales: pd.DataFrame = df_sales_pd[df_sales_pd["category"] == "Electronics"]
    print(f"Number of Electronics Sales: {len(electronics_sales)}")
    display(electronics_sales.head())
    ```

=== "SQL"

    ```python
    # Filter sales for a specific category
    electronics_sales_sql = """
        SELECT *
        FROM sales
        WHERE category = 'Electronics'
    """
    electronics_sales: pd.DataFrame = pd.read_sql(electronics_sales_sql, conn)
    print(f"Number of Electronics Sales: {len(electronics_sales)}")
    display(pd.read_sql(electronics_sales_sql + "LIMIT 5", conn))
    ```

=== "PySpark"

    ```python
    # Filter sales data for specific category
    electronics_sales: psDataFrame = df_sales_ps.filter(
        df_sales_ps["category"] == "Electronics"
    )
    print(f"Number of Electronics Sales: {electronics_sales.count()}")
    electronics_sales.show(10)
    ```

=== "Polars"

    ```python
    # Filter sales data for specific category
    electronics_sales = df_sales_pl.filter(df_sales_pl["category"] == "Electronics")
    print(f"Number of Electronics Sales: {len(electronics_sales)}")
    display(electronics_sales.head(10))
    ```

We can also use numerical filtering, as you can see in the next example, where we filter for sales amounts greater than $500.

=== "Pandas"

    ```python
    # Filter for high value transactions (over $500)
    high_value_sales: pd.DataFrame = df_sales_pd[df_sales_pd["sales_amount"] > 500]
    print(f"Number of high-value Sales: {len(high_value_sales)}")
    display(high_value_sales.head())
    ```

=== "SQL"

    ```python
    # Filter for high value transactions (over $500)
    high_value_sales_sql = """
        SELECT *
        FROM sales
        WHERE sales_amount > 500
    """
    high_value_sales: pd.DataFrame = pd.read_sql(high_value_sales_sql, conn)
    print(f"Number of high-value Sales: {len(high_value_sales)}")
    display(pd.read_sql(high_value_sales_sql + "LIMIT 5", conn))
    ```

=== "PySpark"

    ```python
    # Filter for high value transactions (over $500)
    high_value_sales: psDataFrame = df_sales_ps.filter("sales_amount > 500")
    print(f"Number of high-value Sales: {high_value_sales.count()}")
    high_value_sales.show(10)
    ```

=== "Polars"

    ```python
    # Filter for high value transactions (over $500)
    high_value_sales = df_sales_pl.filter(df_sales_pl["sales_amount"] > 500)
    print(f"Number of high-value Sales: {len(high_value_sales)}")
    display(high_value_sales.head(10))
    ```

When it comes to selecting specific columns, we can use the double square brackets syntax to specify the columns we want to keep in the DataFrame. This allows us to create a new DataFrame with only the relevant columns.

=== "Pandas"

    ```python
    # Select specific columns
    sales_summary: pd.DataFrame = df_sales_pd[["date", "category", "sales_amount"]]
    print(f"Sales Summary DataFrame: {len(sales_summary)}")
    display(sales_summary.head())
    ```

=== "SQL"

    ```python
    # Select specific columns
    sales_summary_sql = """
        SELECT date, category, sales_amount
        FROM sales
    """
    sales_summary: pd.DataFrame = pd.read_sql(sales_summary_sql, conn)
    print(f"Selected columns in Sales: {len(sales_summary)}")
    display(pd.read_sql(sales_summary_sql + "LIMIT 5", conn))
    ```

=== "PySpark"

    ```python
    # Select specific columns
    sales_summary: psDataFrame = df_sales_ps.select("date", "category", "sales_amount")
    print(f"Sales Summary DataFrame: {sales_summary.count()}")
    sales_summary.show(10)
    ```

=== "Polars"

    ```python
    # Select specific columns
    sales_summary = df_sales_pl.select(["date", "category", "sales_amount"])
    print(f"Sales Summary DataFrame: {len(sales_summary)}")
    display(sales_summary.head(10))
    ```

## 2. Grouping and Aggregation

The second section will cover grouping and aggregation techniques. These operations are essential for summarizing data and extracting insights from large datasets.

=== "Pandas"

    In Pandas, we can use the [`.agg()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.agg.html) method to perform aggregation operations on DataFrames. This method allows us to apply multiple aggregation functions to different columns in a single operation.

    ```python
    # Basic aggregation
    sales_stats: pd.DataFrame = df_sales_pd.agg(
        {
            "sales_amount": ["sum", "mean", "min", "max", "count"],
            "quantity": ["sum", "mean", "min", "max"],
        }
    )
    print(f"Sales Statistics: {len(sales_stats)}")
    display(sales_stats)
    ```

=== "SQL"

    ```python
    # Basic aggregation
    sales_stats_sql = """
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
    print(f"Sales Statistics: {len(pd.read_sql(sales_stats_sql, conn))}")
    display(pd.read_sql(sales_stats_sql, conn))
    ```

=== "PySpark"

    ```python
    # Basic aggregation
    sales_stats: psDataFrame = df_sales_ps.agg(
        F.sum("sales_amount").alias("sales_sum"),
        F.avg("sales_amount").alias("sales_mean"),
        F.expr("MIN(sales_amount) AS sales_min"),
        F.expr("MAX(sales_amount) AS sales_max"),
        F.count("*").alias("sales_count"),
        F.expr("SUM(quantity) AS quantity_sum"),
        F.expr("AVG(quantity) AS quantity_mean"),
        F.min("quantity").alias("quantity_min"),
        F.max("quantity").alias("quantity_max"),
    )
    print(f"Sales Statistics: {sales_stats.count()}")
    sales_stats.show()
    ```

=== "Polars"

    ```python
    # Basic aggregation
    sales_stats = df_sales_pl.select(
        pl.col("sales_amount").sum().alias("sales_sum"),
        pl.col("sales_amount").mean().alias("sales_mean"),
        pl.col("sales_amount").min().alias("sales_min"),
        pl.col("sales_amount").max().alias("sales_max"),
        pl.col("quantity").sum().alias("quantity_sum"),
        pl.col("quantity").mean().alias("quantity_mean"),
        pl.col("quantity").min().alias("quantity_min"),
        pl.col("quantity").max().alias("quantity_max"),
    )
    print(f"Sales Statistics: {len(sales_stats)}")
    display(sales_stats)
    ```

It is also possible to group data by a specific column and then apply aggregation functions to summarize the data.

=== "Pandas"

    This is done using the [`.groupby()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html) method to group data by one or more columns and then apply aggregation functions to summarize the data, followed by the [`.agg()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.agg.html) method.

    ```python
    # Group by category and aggregate
    category_sales: pd.DataFrame = df_sales_pd.groupby("category").agg(
        {
            "sales_amount": ["sum", "mean", "count"],
            "quantity": "sum",
        }
    )
    print(f"Category Sales Summary: {len(category_sales)}")
    display(category_sales)
    ```

=== "SQL"

    ```python
    # Group by category and aggregate
    category_sales_sql = """
        SELECT
            category,
            SUM(sales_amount) AS total_sales,
            AVG(sales_amount) AS average_sales,
            COUNT(*) AS transaction_count,
            SUM(quantity) AS total_quantity
        FROM sales
        GROUP BY category
    """
    print(f"Category Sales Summary: {len(pd.read_sql(category_sales_sql, conn))}")
    display(pd.read_sql(category_sales_sql + "LIMIT 5", conn))
    ```

=== "PySpark"

    ```python
    # Group by category and aggregate
    category_sales: psDataFrame = df_sales_ps.groupBy("category").agg(
        F.sum("sales_amount").alias("total_sales"),
        F.avg("sales_amount").alias("average_sales"),
        F.count("*").alias("transaction_count"),
        F.sum("quantity").alias("total_quantity"),
    )
    print(f"Category Sales Summary: {category_sales.count()}")
    category_sales.show()
    ```

=== "Polars"

    ```python
    # Group by category and aggregate
    category_sales = df_sales_pl.group_by("category").agg(
        pl.col("sales_amount").sum().alias("total_sales"),
        pl.col("sales_amount").mean().alias("average_sales"),
        pl.col("sales_amount").count().alias("transaction_count"),
        pl.col("quantity").sum().alias("total_quantity"),
    )
    print(f"Category Sales Summary: {len(category_sales)}")
    display(category_sales.head(10))
    ```

We can rename the columns for clarity by simply assigning new names.

=== "Pandas"

    In Pandas, we use the [`.columns`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.columns.html) attribute of the DataFrame. This makes it easier to understand the results of the aggregation.

    ```python
    # Rename columns for clarity
    category_sales.columns = [
        "total_sales",
        "average_sales",
        "transaction_count",
        "total_quantity",
    ]
    print(f"Renamed Category Sales Summary: {len(category_sales)}")
    display(category_sales.head(10))
    ```

=== "SQL"

    ```python
    ```

=== "PySpark"

    ```python
    # Rename columns for clarity
    category_sales = category_sales.withColumnsRenamed(
        {
            "total_sales": "Total Sales",
            "average_sales": "Average Sales",
            "transaction_count": "Transaction Count",
            "total_quantity": "Total Quantity",
        }
    )
    print(f"Renamed Category Sales Summary: {category_sales.count()}")
    category_sales.show()
    ```

=== "Polars"

    ```python
    # Rename columns for clarity
    category_sales = category_sales.rename(
        {
            "total_sales": "Total Sales",
            "average_sales": "Average Sales",
            "transaction_count": "Transaction Count",
            "total_quantity": "Total Quantity",
        }
    )
    print(f"Renamed Category Sales Summary: {len(category_sales)}")
    display(category_sales.head(10))
    ```

Having aggregated the data, we can now visualize the results using [Plotly](https://plotly.com/python/). This allows us to create interactive visualizations that can help us better understand the data.

=== "Pandas"

    ```python
    # Plot the results
    fig: go.Figure = px.bar(
        category_sales.reset_index(),
        x="category",
        y="total_sales",
        title="Total Sales by Category",
        text="transaction_count",
        labels={"total_sales": "Total Sales ($)", "category": "Product Category"},
    )
    fig.show()
    ```

=== "SQL"

    ```python
    # Plot the results
    fig: go.Figure = px.bar(
        pd.read_sql(category_sales_sql, conn),
        x="category",
        y="total_sales",
        title="Total Sales by Category",
        text="transaction_count",
        labels={"total_sales": "Total Sales ($)", "category": "Product Category"},
    )
    fig.show()
    ```

=== "PySpark"

    ```python
    # Convert to pandas for plotting with plotly
    category_sales_pd: pd.DataFrame = category_sales.toPandas()
    fig: go.Figure = px.bar(
        category_sales_pd,
        x="category",
        y="Total Sales",
        title="Total Sales by Category",
        text="Transaction Count",
        labels={"Total Sales": "Total Sales ($)", "category": "Product Category"},
    )
    fig.show()
    ```

=== "Polars"

    ```python
    # Plot the results
    fig: go.Figure = px.bar(
        category_sales,
        x="category",
        y="Total Sales",
        title="Total Sales by Category",
        text="Transaction Count",
        labels={"Total Sales": "Total Sales ($)", "category": "Product Category"},
    )
    fig.show()
    ```

## 3. Joining

The third section will demonstrate how to join DataFrames to combine data from different sources. This is a common operation in data analysis, allowing us to enrich our data with additional information.

Here, we will join the `sales` DataFrame with the `product` DataFrame to get additional information about the products sold.

=== "Pandas"

    In Pandas, we can use the [`pd.merge()`](https://pandas.pydata.org/docs/reference/api/pandas.merge.html) method to join DataFrames. This method allows us to specify the columns to join on and the type of join (inner, outer, left, or right).

    ```python
    # Join sales with product data
    sales_with_product: pd.DataFrame = pd.merge(
        left=df_sales_pd,
        right=df_product_pd[["product_id", "product_name", "price"]],
        on="product_id",
        how="left",
    )
    print(f"Sales with Product Information: {len(sales_with_product)}")
    display(sales_with_product.head())
    ```

=== "SQL"

    ```python
    # Join sales with product data
    sales_with_product_sql = """
        SELECT s.*, p.product_name, p.price
        FROM sales s
        LEFT JOIN product p ON s.product_id = p.product_id
    """
    print(f"Sales with Product Data: {len(pd.read_sql(sales_with_product_sql, conn))}")
    display(pd.read_sql(sales_with_product_sql + "LIMIT 5", conn))
    ```

=== "PySpark"

    ```python
    # Join sales with product data
    sales_with_product: psDataFrame = df_sales_ps.join(
        other=df_product_ps.select("product_id", "product_name", "price"),
        on="product_id",
        how="left",
    )
    print(f"Sales with Product Information: {sales_with_product.count()}")
    sales_with_product.show(10)
    ```

=== "Polars"

    ```python
    # Join sales with product data
    sales_with_product = df_sales_pl.join(
        df_product_pl.select(["product_id", "product_name", "price"]),
        on="product_id",
        how="left",
    )
    print(f"Sales with Product Information: {len(sales_with_product)}")
    display(sales_with_product.head(10))
    ```

In the next step, we will join the resulting DataFrame with the `customer` DataFrame to get customer information for each sale. This allows us to create a complete view of the sales data, including product and customer details.

=== "Pandas"

    ```python
    # Join with customer information to get a complete view
    complete_sales: pd.DataFrame = pd.merge(
        sales_with_product,
        df_customer_pd[["customer_id", "customer_name", "city", "state"]],
        on="customer_id",
        how="left",
    )
    print(f"Complete Sales Data with Customer Information: {len(complete_sales)}")
    display(complete_sales.head())
    ```

=== "SQL"

    ```python
    # Join with customer information to get a complete view
    complete_sales_sql = """
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
    print(f"Complete Sales Data: {len(pd.read_sql(complete_sales_sql, conn))}")
    display(pd.read_sql(complete_sales_sql + "LIMIT 5", conn))
    ```

=== "PySpark"

    ```python
    # Join with customer information to get a complete view
    complete_sales: psDataFrame = sales_with_product.alias("s").join(
        other=df_customer_ps.select("customer_id", "customer_name", "city", "state").alias(
            "c"
        ),
        on="customer_id",
        how="left",
    )
    print(f"Complete Sales Data with Customer Information: {complete_sales.count()}")
    complete_sales.show(10)
    ```

=== "Polars"

    ```python
    # Join with customer information to get a complete view
    complete_sales = sales_with_product.join(
        df_customer_pl.select(["customer_id", "customer_name", "city", "state"]),
        on="customer_id",
        how="left",
    )
    print(f"Complete Sales Data with Customer Information: {len(complete_sales)}")
    display(complete_sales.head(10))
    ```

Once we have the complete sales data, we can calculate the revenue for each sale by multiplying the price and quantity (columns from different tables). We can also compare this calculated revenue with the sales amount to identify any discrepancies.

=== "Pandas"

    ```python
    # Calculate revenue (price * quantity) and compare with sales amount
    complete_sales["calculated_revenue"] = (
        complete_sales["price"] * complete_sales["quantity"]
    )
    complete_sales["price_difference"] = (
        complete_sales["sales_amount"] - complete_sales["calculated_revenue"]
    )
    print(
        f"Complete Sales Data with Calculated Revenue and Price Difference: {len(complete_sales)}"
    )
    display(
        complete_sales[
            ["sales_amount", "price", "quantity", "calculated_revenue", "price_difference"]
        ].head()
    )
    ```

=== "SQL"

    ```python
    # Calculate revenue and price difference
    revenue_comparison_sql = """
        SELECT
            s.sales_amount,
            p.price,
            s.quantity,
            (p.price * s.quantity) AS calculated_revenue,
            (s.sales_amount - (p.price * s.quantity)) AS price_difference
        FROM sales s
        LEFT JOIN product p ON s.product_id = p.product_id
    """
    print(f"Revenue Comparison: {len(pd.read_sql(revenue_comparison_sql, conn))}")
    display(pd.read_sql(revenue_comparison_sql + "LIMIT 5", conn))
    ```

=== "PySpark"

    ```python
    # Calculate revenue (price * quantity) and compare with sales amount
    complete_sales = complete_sales.withColumns(
        {
            "calculated_revenue": complete_sales["price"] * complete_sales["quantity"],
            "price_difference": F.expr("sales_amount - (price * quantity)"),
        },
    )
    print(
        f"Complete Sales Data with Calculated Revenue and Price Difference: {complete_sales.count()}"
    )
    complete_sales.select(
        "sales_amount", "price", "quantity", "calculated_revenue", "price_difference"
    ).show(10)
    ```

=== "Polars"

    ```python
    # Calculate revenue (price * quantity) and compare with sales amount
    complete_sales = complete_sales.with_columns(
        (pl.col("price") * pl.col("quantity")).alias("calculated_revenue"),
        (pl.col("sales_amount") - (pl.col("price") * pl.col("quantity"))).alias(
            "price_difference"
        ),
    )
    print(
        f"Complete Sales Data with Calculated Revenue and Price Difference: {len(complete_sales)}"
    )
    display(
        complete_sales.select(
            ["sales_amount", "price", "quantity", "calculated_revenue", "price_difference"]
        ).head(10)
    )
    ```

## 4. Window Functions

Window functions are a powerful feature in Pandas that allow us to perform calculations across a set of rows related to the current row. This is particularly useful for time series data, where we may want to calculate rolling averages, cumulative sums, or other metrics based on previous or subsequent rows.

To understand more about the nuances of the window functions, check out some of these guides:

- [Analyzing data with window functions](https://docs.snowflake.com/en/user-guide/functions-window-using)
- [SQL Window Functions Visualized](https://medium.com/learning-sql/sql-window-function-visualized-fff1927f00f2)

In this section, we will demonstrate how to use window functions to analyze sales data over time. We will start by converting the `date` column to a datetime type, which is necessary for time-based calculations. We will then group the data by date and calculate the total sales for each day.

=== "Pandas"

    ```python
    # Time-based window function
    df_sales_pd["date"] = pd.to_datetime(df_sales_pd["date"])  # Ensure date type
    daily_sales: pd.DataFrame = (
        df_sales_pd.groupby(df_sales_pd["date"].dt.date)["sales_amount"]
        .sum()
        .reset_index()
        .sort_values("date")
    )
    print(f"Daily Sales Summary: {len(daily_sales)}")
    display(daily_sales.head())
    ```

=== "SQL"

    ```python
    # Time-based window function
    daily_sales_sql = """
        SELECT
            date,
            SUM(sales_amount) AS total_sales
        FROM sales
        GROUP BY date
        ORDER BY date
    """
    print(f"Daily Sales Data: {len(pd.read_sql(daily_sales_sql, conn))}")
    display(pd.read_sql(daily_sales_sql + "LIMIT 5", conn))
    ```

=== "PySpark"

    ```python
    # Convert date column to date type if not already
    df_sales_ps = df_sales_ps.withColumn("date", F.to_date(df_sales_ps["date"]))
    daily_sales: psDataFrame = (
        df_sales_ps.groupBy("date")
        .agg(
            F.sum("sales_amount").alias("total_sales"),
        )
        .orderBy("date")
    )
    print(f"Daily Sales Summary: {daily_sales.count()}")
    daily_sales.show(10)
    ```

=== "Polars"

    ```python
    # Convert date column to date type if not already
    df_sales_pl = df_sales_pl.with_columns(pl.col("date").cast(pl.Date))
    ```

    ```python
    daily_sales = (
        df_sales_pl.group_by("date")
        .agg(
            pl.col("sales_amount").sum().alias("total_sales"),
        )
        .sort("date")
    )
    print(f"Daily Sales Summary: {len(daily_sales)}")
    display(daily_sales.head(10))
    ```

Next, we will calculate the rolling average of sales over a 7-day window.

=== "Pandas"

    This is done using the [`.rolling()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html) method, which allows us to specify the window size and the minimum number of periods required for the calculation.

    ```python
    # Calculate rolling averages (7-day moving average)
    daily_sales["7d_moving_avg"] = (
        daily_sales["sales_amount"].rolling(window=7, min_periods=1).mean()
    )
    print(f"Daily Sales with 7-Day Moving Average: {len(daily_sales)}")
    display(daily_sales.head())
    ```

=== "SQL"

    ```python
    # Window functions for lead and lag
    window_sql = """
        SELECT
            date AS sale_date,
            SUM(sales_amount) AS sales_amount,
            AVG(sales_amount) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS rolling_7d_avg,
            LAG(SUM(sales_amount)) OVER (ORDER BY date) AS previous_day_sales,
            LEAD(SUM(sales_amount)) OVER (ORDER BY date) AS next_day_sales,
            SUM(sales_amount) - LAG(SUM(sales_amount)) OVER (ORDER BY date) AS day_over_day_change
        FROM sales
        GROUP BY date
        ORDER BY date
    """
    window_df: pd.DataFrame = pd.read_sql(window_sql, conn)
    print(f"Window Functions: {len(window_df)}")
    display(pd.read_sql(window_sql + "LIMIT 5", conn))
    ```

=== "PySpark"

    ```python
    # Calculate day-over-day change
    daily_sales = daily_sales.withColumns(
        {
            "day_over_day_change": F.expr("total_sales - previous_day_sales"),
            "pct_change": (F.expr("total_sales / previous_day_sales - 1") * 100).alias(
                "pct_change"
            ),
        }
    )
    print(f"Daily Sales with Day-over-Day Change: {daily_sales.count()}")
    daily_sales.show(10)
    ```

=== "Polars"

    ```python
    # Calculate day-over-day change
    daily_sales = daily_sales.with_columns(
        (pl.col("total_sales") - pl.col("previous_day_sales")).alias("day_over_day_change"),
        (pl.col("total_sales") / pl.col("previous_day_sales") - 1).alias("pct_change")
        * 100,
    )
    print(f"Daily Sales with Day-over-Day Change: {len(daily_sales)}")
    display(daily_sales.head(10))
    ```

Next, we will calculate the lag and lead values for the sales amount. This allows us to compare the current day's sales with the previous and next days' sales.

=== "Pandas"

    This is done using the [`.shift()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shift.html) method, which shifts the values in a column by a specified number of periods. Note that the [`.shift()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shift.html) method simply shifts the values in the column, so we can use it to create lag and lead columns. This function itself does not need to be ordered because it assumes that the DataFrame is already ordered; if you want it to be ordered, you can use the [`.sort_values()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html) method before applying [`.shift()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shift.html).

    ```python
    # Calculate lag and lead
    daily_sales["previous_day_sales"] = daily_sales["sales_amount"].shift(1)
    daily_sales["next_day_sales"] = daily_sales["sales_amount"].shift(-1)
    print(f"Daily Sales with Lag and Lead: {len(daily_sales)}")
    display(daily_sales.head())
    ```

=== "SQL"

    ```python
    ```

=== "PySpark"

    ```python
    # Define a window specification for lead/lag functions
    window_spec = Window.orderBy("date")

    # Calculate lead and lag
    daily_sales = daily_sales.withColumns(
        {
            "previous_day_sales": F.lag("total_sales").over(window_spec),
            "next_day_sales": F.expr("LEAD(total_sales) OVER (ORDER BY date)"),
        },
    )
    print(f"Daily Sales with Lead and Lag: {daily_sales.count()}")
    daily_sales.show(10)
    ```

=== "Polars"

    ```python
    # Calculate lead and lag
    daily_sales = daily_sales.with_columns(
        pl.col("total_sales").shift(1).alias("previous_day_sales"),
        pl.col("total_sales").shift(-1).alias("next_day_sales"),
    )
    print(f"Daily Sales with Lead and Lag: {len(daily_sales)}")
    display(daily_sales.head(10))
    ```

Now, we can calculate the day-over-day change in sales. This is done by subtracting the previous day's sales from the current day's sales.

=== "Pandas"

    We can also calculate the percentage change in sales using the [`.pct_change()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pct_change.html) method, which calculates the percentage change between the current and previous values.

    ```python
    # Calculate day-over-day change
    daily_sales["day_over_day_change"] = (
        daily_sales["sales_amount"].pct_change() - daily_sales["previous_day_sales"]
    )
    daily_sales["pct_change"] = daily_sales["sales_amount"].pct_change() * 100
    print(f"Daily Sales with Day-over-Day Change: {len(daily_sales)}")
    display(daily_sales.head())
    ```

=== "SQL"

    ```python
    ```

=== "PySpark"

    ```python
    # Calculate 7-day moving average
    daily_sales = daily_sales.withColumns(
        {
            "7d_moving_avg": F.avg("total_sales").over(
                Window.orderBy("date").rowsBetween(-6, 0)
            ),
            "7d_rolling_avg": F.expr(
                "AVG(total_sales) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)"
            ),
        }
    )
    print(f"Daily Sales with 7-Day Moving Average: {daily_sales.count()}")
    daily_sales.show(10)
    ```

=== "Polars"

    ```python
    # Calculate 7-day moving average
    daily_sales = daily_sales.with_columns(
        pl.col("total_sales")
        .rolling_mean(window_size=7, min_periods=1)
        .alias("7d_moving_avg"),
    )
    print(f"Daily Sales with 7-Day Moving Average: {len(daily_sales)}")
    display(daily_sales.head(10))
    ```

Finally, we can visualize the daily sales data along with the 7-day moving average using Plotly. This allows us to see the trends in sales over time and how the moving average smooths out the fluctuations in daily sales.

=== "Pandas"

    ```python
    # Plot time series with rolling average
    fig = (
        go.Figure()
        .add_trace(
            go.Scatter(
                x=daily_sales["date"],
                y=daily_sales["sales_amount"],
                mode="lines",
                name="Daily Sales",
            )
        )
        .add_trace(
            go.Scatter(
                x=daily_sales["date"],
                y=daily_sales["7d_moving_avg"],
                mode="lines",
                name="7-Day Moving Average",
                line=dict(width=3),
            ),
        )
        .update_layout(
            title="Daily Sales with 7-Day Moving Average",
            xaxis_title="Date",
            yaxis_title="Sales Amount ($)",
        )
    )
    fig.show()
    ```

=== "SQL"

    ```python
    # Plot time series with rolling average
    fig = (
        go.Figure()
        .add_trace(
            go.Scatter(
                x=window_df["sale_date"],
                y=window_df["sales_amount"],
                mode="lines",
                name="Daily Sales",
            )
        )
        .add_trace(
            go.Scatter(
                x=window_df["sale_date"],
                y=window_df["rolling_7d_avg"],
                mode="lines",
                name="7-Day Moving Average",
                line=dict(width=3),
            )
        )
        .update_layout(
            title="Daily Sales with 7-Day Moving Average",
            xaxis_title="Date",
            yaxis_title="Sales Amount ($)",
        )
    )
    fig.show()
    ```

=== "PySpark"

    ```python
    # Plot time series with rolling average
    fig = (
        go.Figure()
        .add_trace(
            go.Scatter(
                x=daily_sales.toPandas()["date"],
                y=daily_sales.toPandas()["total_sales"],
                mode="lines",
                name="Daily Sales",
            )
        )
        .add_trace(
            go.Scatter(
                x=daily_sales.toPandas()["date"],
                y=daily_sales.toPandas()["7d_moving_avg"],
                mode="lines",
                name="7-Day Moving Average",
                line=dict(width=3),
            ),
        )
        .update_layout(
            title="Daily Sales with 7-Day Moving Average",
            xaxis_title="Date",
            yaxis_title="Sales Amount ($)",
        )
    )
    fig.show()
    ```

=== "Polars"

    ```python
    # Plot time series with rolling average
    fig = (
        go.Figure()
        .add_trace(
            go.Scatter(
                x=daily_sales["date"].to_list(),
                y=daily_sales["total_sales"].to_list(),
                mode="lines",
                name="Daily Sales",
            )
        )
        .add_trace(
            go.Scatter(
                x=daily_sales["date"].to_list(),
                y=daily_sales["7d_moving_avg"].to_list(),
                mode="lines",
                name="7-Day Moving Average",
                line=dict(width=3),
            )
        )
        .update_layout(
            title="Daily Sales with 7-Day Moving Average",
            xaxis_title="Date",
            yaxis_title="Sales Amount ($)",
        )
    )
    fig.show()
    ```

## 5. Ranking and Partitioning

The fifth section will demonstrate how to rank and partition data in Pandas. This is useful for identifying top performers, such as the highest spending customers or the most popular products.

=== "Pandas"

    In Pandas, we can use the [`.rank()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rank.html) method to rank values in a DataFrame. This method allows us to specify the ranking method (e.g., dense, average, min, max) and whether to rank in ascending or descending order.

    ```python
    # Rank customers by total spending
    customer_spending: pd.DataFrame = (
        df_sales_pd.groupby("customer_id")["sales_amount"].sum().reset_index()
    )
    customer_spending["rank"] = customer_spending["sales_amount"].rank(
        method="dense", ascending=False
    )
    customer_spending = customer_spending.sort_values("rank")
    print(f"Customer Spending Summary: {len(customer_spending)}")
    display(customer_spending.head(10))
    ```

=== "SQL"

    ```python
    ```

=== "PySpark"

    ```python
    # Rank customers by total spending
    customer_spending: psDataFrame = (
        df_sales_ps.groupBy("customer_id")
        .agg(F.sum("sales_amount").alias("total_spending"))
        .withColumn("rank", F.dense_rank().over(Window.orderBy(F.desc("total_spending"))))
        .orderBy("rank")
    )
    print(f"Customer Spending Summary: {customer_spending.count()}")
    customer_spending.show(10)
    ```

=== "Polars"

    ```python
    # Rank customers by total spending
    customer_spending = (
        df_sales_pl.group_by("customer_id")
        .agg(pl.col("sales_amount").sum().alias("total_spending"))
        .with_columns(
            pl.col("total_spending").rank(method="dense", descending=True).alias("rank")
        )
        .sort("rank")
    )
    print(f"Customer Spending Summary: {len(customer_spending)}")
    display(customer_spending.head(10))
    ```

Once we have ranked the customers, we can merge this information with the `customer` DataFrame to get additional details about each customer, such as their name, segment, and city.

=== "Pandas"

    ```python
    # Add customer details
    top_customers: pd.DataFrame = pd.merge(
        customer_spending,
        df_customer_pd[["customer_id", "customer_name", "segment", "city"]],
        on="customer_id",
        how="left",
    )
    print(f"Top Customers Summary: {len(top_customers)}")
    display(top_customers.head(10))
    ```

=== "SQL"

    ```python
    # Rank customers by total spending
    customer_spending_sql = """
        SELECT
            c.customer_id,
            c.customer_name,
            c.segment,
            c.city,
            SUM(s.sales_amount) AS total_spending,
            RANK() OVER (ORDER BY SUM(s.sales_amount) DESC) AS rank
        FROM sales s
        JOIN customer c ON s.customer_id = c.customer_id
        GROUP BY c.customer_id, c.customer_name, c.segment, c.city
        ORDER BY rank
    """
    print(f"Customer Spending: {len(pd.read_sql(customer_spending_sql, conn))}")
    display(pd.read_sql(customer_spending_sql + "LIMIT 10", conn))
    ```

=== "PySpark"

    ```python
    ```

=== "Polars"

    ```python
    ```

Next, we will rank products based on the quantity sold. This allows us to identify the most popular products in terms of sales volume.

=== "Pandas"

    ```python
    # Rank products by quantity sold
    product_popularity: pd.DataFrame = (
        df_sales_pd.groupby("product_id")["quantity"].sum().reset_index()
    )
    product_popularity["rank"] = product_popularity["quantity"].rank(
        method="dense", ascending=False
    )
    product_popularity = product_popularity.sort_values("rank")
    print(f"Product Popularity Summary: {len(product_popularity)}")
    display(product_popularity.head(10))
    ```

=== "SQL"

    ```python
    # Rank products by quantity sold
    product_popularity_sql = """
        SELECT
            p.product_id,
            p.product_name,
            p.category,
            SUM(s.quantity) AS total_quantity,
            RANK() OVER (ORDER BY SUM(s.quantity) DESC) AS rank
        FROM sales s
        JOIN product p ON s.product_id = p.product_id
        GROUP BY p.product_id, p.product_name, p.category
        ORDER BY rank
    """
    print(f"Product Popularity: {len(pd.read_sql(product_popularity_sql, conn))}")
    display(pd.read_sql(product_popularity_sql + "LIMIT 10", conn))
    ```

=== "PySpark"

    ```python
    # Rank products by quantity sold
    product_popularity: psDataFrame = (
        df_sales_ps.groupBy("product_id")
        .agg(F.sum("quantity").alias("total_quantity"))
        .withColumn("rank", F.expr("DENSE_RANK() OVER (ORDER BY total_quantity DESC)"))
        .orderBy("rank")
    )
    print(f"Product Popularity Summary: {product_popularity.count()}")
    product_popularity.show(10)
    ```

=== "Polars"

    ```python
    # Rank products by quantity sold
    product_popularity = (
        df_sales_pl.group_by("product_id")
        .agg(pl.col("quantity").sum().alias("total_quantity"))
        .with_columns(
            pl.col("total_quantity").rank(method="dense", descending=True).alias("rank")
        )
        .sort("rank")
    )
    print(f"Product Popularity Summary: {len(product_popularity)}")
    display(product_popularity.head(10))
    ```

As with the customer data, we can merge the product popularity information with the `product` DataFrame to get additional details about each product, such as its name and category.

=== "Pandas"

    ```python
    # Add product details
    top_products: pd.DataFrame = pd.merge(
        product_popularity,
        df_product_pd[["product_id", "product_name", "category"]],
        on="product_id",
        how="left",
    )
    print(f"Top Products Summary: {len(top_products)}")
    display(top_products.head(10))
    ```

=== "SQL"

    ```python
    ```

=== "PySpark"

    ```python
    ```

=== "Polars"

    ```python
    ```


## Conclusion

    ```python
    ```
