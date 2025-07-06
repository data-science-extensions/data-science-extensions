# Data Querying for Data Scientists


### A Comprehensive Guide of using Pandas, SQL, PySpark, and Polars for Data Manipulation Techniques, with Practical Examples and Visualisations

[![](./../../assets/images/dse.png){width=70px}<br>Download<br>- ALL](./index-r.ipynb){ :download .md-button .md-button-fifth }
[![](./../../assets/icons/pandas.svg){width=100%}<br>Download<br>- Pandas](./index-pandas-r.ipynb){ :download .md-button .md-button-fifth }
[![](./../../assets/icons/sql.svg){width=100%}<br>Download<br>- SQL](./index-sql-r.ipynb){ :download .md-button .md-button-fifth }
[![](./../../assets/icons/spark.svg){width=100%}<br>Download<br>- PySpark](./index-pyspark-r.ipynb){ :download .md-button .md-button-fifth }
[![](./../../assets/icons/polars.svg){width=100%}<br>Download<br>- Polars](./index-polars-r.ipynb){ :download .md-button .md-button-fifth }


## Introduction

Working as a Data Scientist or Data Engineer often involves querying data from various sources. There are many tools and libraries available to perform these tasks, each with its own strengths and weaknesses. Also, there are many different ways to achieve similar results, depending on the tool or library used. It's important to be familiar with these different methods to choose the best one for your specific use case.

This article provides a comprehensive guide on how to query data using different tools and libraries, including Pandas, SQL, PySpark, and Polars. Each section will cover the setup, data creation, and various querying techniques such as filtering, grouping, joining, window functions, ranking, and sorting. The output will be identical across all tools, but the transformations will be implemented using the specific syntax and features of each library. Therefore allowing you to compare the different approaches and understand the nuances of each method.


## Overview of the Different Libraries

Before we dive into the querying techniques, let's take a moment to understand the different libraries and tools we will be using in this article. Each library has its own strengths and weaknesses, and understanding these can help you choose the right tool for your specific use case.

Throughout this article, you can easily switch between the different libraries by selecting the appropriate tab. Each section will provide the same functionality, but implemented using the specific syntax and features of each library.

### Pandas

[Pandas] is a powerful data manipulation library in Python that provides data structures and functions for working with structured data. It is widely used for data analysis and manipulation tasks.

Historically, Pandas was one of the first libraries to provide a DataFrame structure, which is similar to a table in a relational database. It allows for easy data manipulation, filtering, grouping, and aggregation. Pandas is built on top of [NumPy][numpy] and provides a high-level interface for working with data. It is particularly well-suited for small to medium-sized datasets and is often used in Data Science and Machine Learning workflows.

Pandas provides a rich set of functionalities for data manipulation, including filtering, grouping, joining, and window functions. It also integrates well with other libraries such as Matplotlib and Seaborn for data visualization, making it a popular choice among data scientists and analysts.

While Pandas is both powerful and popular, it is important to note that it operates **in-memory**, which means that it may not be suitable for very large datasets that do not fit into memory. In such cases, other libraries like PySpark or Polars may be more appropriate.

### SQL

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

### PySpark

[PySpark] is the Python API for Apache Spark, a distributed computing framework that allows for large-scale data processing. PySpark provides a high-level interface for working with Spark, making it easier to write distributed data processing applications in Python. It is particularly well-suited for big data processing and analytics.

PySpark provides a DataFrame API similar to Pandas, but it is designed to work with large datasets that do not fit into memory. It allows for distributed data processing across a cluster of machines, making it suitable for big data applications. PySpark supports various data sources, including [HDFS], [S3], [ADLS], and [JDBC], and provides powerful features for filtering, grouping, joining, and aggregating data.

While PySpark is a powerful tool for big data processing, it can be more complex to set up and use compared to Pandas. It requires a Spark cluster and may have a steeper learning curve for those unfamiliar with distributed computing concepts. However, it is an excellent choice for processing large datasets and performing complex data transformations.

### Polars

[Polars] is a fast DataFrame library for Python that is designed for high-performance data manipulation. It is built on top of Rust and provides a DataFrame API similar to Pandas, but with a focus on performance and memory efficiency. Polars is particularly well-suited for large datasets and complex queries.

Polars supports lazy evaluation, which allows for optimizations in query execution. Polars also provides powerful features for filtering, grouping, joining, and aggregating data, making it a great choice for data analysis tasks.

While Polars is a relatively new library compared to Pandas, it has gained popularity for its performance and ease of use. It is designed to be a drop-in replacement for Pandas, allowing users to leverage its performance benefits without significant changes to their existing code. It is particularly useful for data scientists and analysts who need to work with large datasets and require fast data manipulation capabilities. The setup is simple and straightforward, similar to Pandas, and less complex than PySpark. It is a great choice for data analysis tasks that require high performance and memory efficiency.


## Setup

Before we start querying data, we need to set up our environment. This includes importing the necessary libraries, creating sample data, and defining constants that will be used throughout the article. The following sections will guide you through this setup process. The code for this article is also available on GitHub: [querying-data](...) {==**UPDATE URL**==}.

### Pandas

```python {.pandas linenums="1" title="Setup"}
# StdLib Imports
from typing import Any

# Third Party Imports
import numpy as np
import pandas as pd
from plotly import express as px, graph_objects as go, io as pio


# Set seed for reproducibility
np.random.seed(42)

# Determine the number of records to generate
n_records = 100

# Set default Plotly template
pio.templates.default = "simple_white+gridon"

# Set Pandas display options
pd.set_option("display.max_columns", None)
```

### SQL

```python {.sql linenums="1" title="Setup"}
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

### PySpark

```python {.pyspark linenums="1" title="Setup"}
# StdLib Imports
from typing import Any

# Third Party Imports
import numpy as np
from plotly import express as px, graph_objects as go, io as pio
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

# Set default Plotly template
pio.templates.default = "simple_white+gridon"
```

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

### Pandas

To create the dataframes in Pandas, we will use the data we generated earlier. We will parse the dictionaries into Pandas DataFrames, which will allow us to perform various data manipulation tasks.

```python {.pandas linenums="1" title="Create DataFrames"}
df_sales_pd: pd.DataFrame = pd.DataFrame(sales_data)
df_product_pd: pd.DataFrame = pd.DataFrame(product_data)
df_customer_pd: pd.DataFrame = pd.DataFrame(customer_data)
```

Once the data is created, we can check that it has been loaded correctly by displaying the first few rows of each DataFrame. To do this, we will use the [`.head()`][pandas-head] method to display the first 5 rows of each DataFrame, and then parse to the [`print()`][python-print] function to display the DataFrame in a readable format.

```python {.pandas linenums="1" title="Check Sales DataFrame"}
print(f"Sales DataFrame: {len(df_sales_pd)}")
print(df_sales_pd.head(5))
print(df_sales_pd.head(5).to_markdown())
```

<div class="result" markdown>

```txt
Sales DataFrame: 100
```

```txt
        date  customer_id  product_id     category  sales_amount  quantity
0 2023-01-01           52          45         Food        490.76         7
1 2023-01-02           93          41  Electronics        453.94         5
2 2023-01-03           15          29         Home        994.51         5
3 2023-01-04           72          15  Electronics        184.17         7
4 2023-01-05           61          45         Food         27.89         9
```

|      | date                | customer_id | product_id | category    | sales_amount | quantity |
| ---: | :------------------ | ----------: | ---------: | :---------- | -----------: | -------: |
|    0 | 2023-01-01 00:00:00 |          52 |         45 | Food        |       490.76 |        7 |
|    1 | 2023-01-02 00:00:00 |          93 |         41 | Electronics |       453.94 |        5 |
|    2 | 2023-01-03 00:00:00 |          15 |         29 | Home        |       994.51 |        5 |
|    3 | 2023-01-04 00:00:00 |          72 |         15 | Electronics |       184.17 |        7 |
|    4 | 2023-01-05 00:00:00 |          61 |         45 | Food        |        27.89 |        9 |

</div>

```python {.pandas linenums="1" title="Check Product DataFrame"}
print(f"Product DataFrame: {len(df_product_pd)}")
print(df_product_pd.head(5))
print(df_product_pd.head(5).to_markdown())
```

<div class="result" markdown>

```txt
Product DataFrame: 50
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

```python {.pandas linenums="1" title="Check Customer DataFrame"}
print(f"Customer DataFrame: {len(df_customer_pd)}")
print(df_customer_pd.head(5))
print(df_customer_pd.head(5).to_markdown())
```

<div class="result" markdown>

```txt
Customer DataFrame: 100
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

### SQL

To create the dataframes in SQL, we will use the data we generated earlier. Firstly, we need to create the SQLite database. This will be an in-memory database for demonstration purposes, but in a real-world scenario, you would typically connect to a persistent (on-disk) database. To do this, we will use the [`sqlite3`][sqlite3] library to create a connection to the database, which we define with the `:memory:` parameter on the [`.connect()`][sqlite3-connect] function. The result is to create a temporary database that exists only during the lifetime of the connection.

Next, we will then parse the dictionaries into Pandas DataFrames, which will then be loaded into an SQLite database. This allows us to perform various data manipulation tasks using SQL queries.

```python {.sql linenums="1" title="Create DataFrames"}
# Creates SQLite database and tables
conn: sqlite3.Connection = sqlite3.connect(":memory:")
pd.DataFrame(sales_data).to_sql("sales", conn, index=False, if_exists="replace")
pd.DataFrame(product_data).to_sql("product", conn, index=False, if_exists="replace")
pd.DataFrame(customer_data).to_sql("customer", conn, index=False, if_exists="replace")
```

Once the data is created, we can check that it has been loaded correctly by displaying the first few rows of each DataFrame. To do this, we will use the [`pd.read_sql()`][pandas-read_sql] function to execute SQL queries and retrieve the data from the database. We will then parse the results to the [`print()`][python-print] function to display the DataFrame in a readable format.

```python {.sql linenums="1" title="Check Sales DataFrame"}
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

```python {.sql linenums="1" title="Check Product DataFrame"}
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

```python {.sql linenums="1" title="Check Customer DataFrame"}
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

### PySpark

Spark DataFrames are similar to Pandas DataFrames, but they are designed to work with large datasets that do not fit into memory. They can be distributed across a cluster of machines, allowing for parallel processing of data.

To create the dataframes in PySpark, we will use the data we generated earlier. We will first create a Spark session, which is the entry point to using PySpark. Then, we will parse the dictionaries into PySpark DataFrames, which will allow us to perform various data manipulation tasks.

The PySpark session is created using the [`.builder`][pyspark-builder] method on the [`SparkSession`][pyspark-sparksession] class, which allows us to configure the session with various options such as the application name. The [`.getOrCreate()`][pyspark-getorcreate] method is used to either get an existing session or create a new one if it doesn't exist.

```python {.pyspark linenums="1" title="Create Spark Session"}
spark: SparkSession = SparkSession.builder.appName("SalesAnalysis").getOrCreate()
```

Once the Spark session is created, we can create the DataFrames from the dictionaries. We will  use the [`.createDataFrame()`][pyspark-createdataframe] method on the Spark session to convert the dictionaries into PySpark DataFrames. The [`.createDataFrame()`][pyspark-createdataframe] method is expecting the data to be oriented by _row_. Meaning that the data should be in the form of a list of dictionaries, where each dictionary represents a row of data. However, we currently have our data is oriented by _column_, where the dictionarieshave keys as column names and values as lists of data. Therefore, we will first need to convert the dictionaries from _column_ orientation to _row_ orientation. The easiest way to do this is by parse'ing the data to a Pandas DataFrames, and then using that to create our PySpark DataFrames from there.

A good description of how to create PySpark DataFrames from Python Dictionaries can be found in the PySpark documentation: [PySpark Create DataFrame From Dictionary][pyspark-create-dataframe-from-dict].

```python {.pyspark linenums="1" title="Create DataFrames"}
df_sales_ps: psDataFrame = spark.createDataFrame(pd.DataFrame(sales_data))
df_product_ps: psDataFrame = spark.createDataFrame(pd.DataFrame(product_data))
df_customer_ps: psDataFrame = spark.createDataFrame(pd.DataFrame(customer_data))
```

Once the data is created, we can check that it has been loaded correctly by displaying the first few rows of each DataFrame. To do this, we will use the [`.show()`][pyspark-show] method to display the first `5` rows of each DataFrame. The [`.show()`][pyspark-show] method is used to display the data in a tabular format, similar to how it would be displayed in a SQL database.

```python {.pyspark linenums="1" title="Check Sales DataFrame"}
print(f"Sales DataFrame: {df_sales_ps.count()}")
df_sales_ps.show(5)
print(df_sales_ps.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt
Sales DataFrame: 100
```

```txt
+-------------------+-----------+----------+-----------+------------+--------+
|               date|customer_id|product_id|   category|sales_amount|quantity|
+-------------------+-----------+----------+-----------+------------+--------+
|2023-01-01 00:00:00|         52|        45|       Food|      490.76|       7|
|2023-01-02 00:00:00|         93|        41|Electronics|      453.94|       5|
|2023-01-03 00:00:00|         15|        29|       Home|      994.51|       5|
|2023-01-04 00:00:00|         72|        15|Electronics|      184.17|       7|
|2023-01-05 00:00:00|         61|        45|       Food|       27.89|       9|
+-------------------+-----------+----------+-----------+------------+--------+
only showing top 10 rows
```

|      | date                | customer_id | product_id | category    | sales_amount | quantity |
| ---: | :------------------ | ----------: | ---------: | :---------- | -----------: | -------: |
|    0 | 2023-01-01 00:00:00 |          52 |         45 | Food        |       490.76 |        7 |
|    1 | 2023-01-02 00:00:00 |          93 |         41 | Electronics |       453.94 |        5 |
|    2 | 2023-01-03 00:00:00 |          15 |         29 | Home        |       994.51 |        5 |
|    3 | 2023-01-04 00:00:00 |          72 |         15 | Electronics |       184.17 |        7 |
|    4 | 2023-01-05 00:00:00 |          61 |         45 | Food        |        27.89 |        9 |

</div>

```python {.pyspark linenums="1" title="Check Product DataFrame"}
print(f"Product DataFrame: {df_product_ps.count()}")
df_product_ps.show(5)
print(df_product_ps.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt
Product DataFrame: 50
```

```txt
+----------+------------+------+--------+-----------+
|product_id|product_name| price|category|supplier_id|
+----------+------------+------+--------+-----------+
|         1|   Product 1|257.57|    Food|          8|
|         2|   Product 2|414.96|Clothing|          5|
|         3|   Product 3|166.82|Clothing|          8|
|         4|   Product 4|448.81|    Food|          4|
|         5|   Product 5|200.71|    Food|          8|
+----------+------------+------+--------+-----------+
only showing top 5 rows
```

|      | product_id | product_name |  price | category | supplier_id |
| ---: | ---------: | :----------- | -----: | :------- | ----------: |
|    0 |          1 | Product 1    | 257.57 | Food     |           8 |
|    1 |          2 | Product 2    | 414.96 | Clothing |           5 |
|    2 |          3 | Product 3    | 166.82 | Clothing |           8 |
|    3 |          4 | Product 4    | 448.81 | Food     |           4 |
|    4 |          5 | Product 5    | 200.71 | Food     |           8 |

</div>

```python {.pyspark linenums="1" title="Check Customer DataFrame"}
print(f"Customer DataFrame: {df_customer_ps.count()}")
df_customer_ps.show(5)
print(df_customer_ps.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt
Customer DataFrame: 100
```

```txt
+-----------+-------------+-----------+-----+-----------+
|customer_id|customer_name|       city|state|    segment|
+-----------+-------------+-----------+-----+-----------+
|          1|   Customer 1|    Phoenix|   NY|  Corporate|
|          2|   Customer 2|    Phoenix|   CA|Home Office|
|          3|   Customer 3|    Phoenix|   NY|Home Office|
|          4|   Customer 4|Los Angeles|   NY|   Consumer|
|          5|   Customer 5|Los Angeles|   IL|Home Office|
+-----------+-------------+-----------+-----+-----------+
only showing top 5 rows
```

|      | customer_id | customer_name | city        | state | segment     |
| ---: | ----------: | :------------ | :---------- | :---- | :---------- |
|    0 |           1 | Customer 1    | Phoenix     | NY    | Corporate   |
|    1 |           2 | Customer 2    | Phoenix     | CA    | Home Office |
|    2 |           3 | Customer 3    | Phoenix     | NY    | Home Office |
|    3 |           4 | Customer 4    | Los Angeles | NY    | Consumer    |
|    4 |           5 | Customer 5    | Los Angeles | IL    | Home Office |

</div>

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

### Pandas

In Pandas, we can use boolean indexing to filter rows based on specific conditions. As you can see in this first example, this looks like using square brackets, within which we define a column and a condition. In the below example, we can use string values to filter categorical data.

For more information about filtering in Pandas, see the [Pandas documentation on filtering][pandas-subsetting].

```python {.pandas linenums="1" title="Filter sales data for specific category"}
electronics_sales_pd: pd.DataFrame = df_sales_pd[df_sales_pd["category"] == "Electronics"]
print(f"Number of Electronics Sales: {len(electronics_sales_pd)}")
print(electronics_sales_pd.head(5))
print(electronics_sales_pd.head(5).to_markdown())
```

<div class="result" markdown>

```txt
Number of Electronics Sales: 28
```

```txt
         date  customer_id  product_id     category  sales_amount  quantity
1  2023-01-02           93          41  Electronics        453.94         5
3  2023-01-04           72          15  Electronics        184.17         7
8  2023-01-09           75           9  Electronics        746.73         2
10 2023-01-11           88           1  Electronics        314.98         9
11 2023-01-12           24          44  Electronics        547.11         8
```

|      | date                | customer_id | product_id | category    | sales_amount | quantity |
| ---: | :------------------ | ----------: | ---------: | :---------- | -----------: | -------: |
|    1 | 2023-01-02 00:00:00 |          93 |         41 | Electronics |       453.94 |        5 |
|    3 | 2023-01-04 00:00:00 |          72 |         15 | Electronics |       184.17 |        7 |
|    8 | 2023-01-09 00:00:00 |          75 |          9 | Electronics |       746.73 |        2 |
|   10 | 2023-01-11 00:00:00 |          88 |          1 | Electronics |       314.98 |        9 |
|   11 | 2023-01-12 00:00:00 |          24 |         44 | Electronics |       547.11 |        8 |

</div>

### SQL

In SQL, we can use the `WHERE` clause to filter rows based on specific conditions. The syntax should be very familiar to anyone who has worked with SQL before. We can use the [`pd.read_sql()`][pandas-read_sql] function to execute SQL queries and retrieve the data from the database. The result is a Pandas DataFrame that contains only the rows that match the specified condition. In the below example, we filter for sales in the "Electronics" category.

For more information about filtering in SQL, see the [SQL WHERE clause documentation][sqlite-where].

```python {.sql linenums="1" title="Filter sales for a specific category"}
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

### PySpark

In PySpark, we can use the [`.filter()`][pyspark-filter] (or the [`.where()`][pyspark-where]) method to filter rows based on specific conditions. This process is effectively doing a boolean indexing operation to filter the DataFrame. The syntax is similar to SQL, where we can specify the condition as a string or using column expressions. In the below example, we filter for sales in the "Electronics" category.

For more information about filtering in PySpark, see the [PySpark documentation on filtering][pyspark-filtering].

```python {.pyspark linenums="1" title="Filter sales for a specific category"}
electronics_sales_ps: psDataFrame = df_sales_ps.filter(df_sales_ps["category"] == "Electronics")
print(f"Number of Electronics Sales: {electronics_sales_ps.count()}")
electronics_sales_ps.show(5)
print(electronics_sales_ps.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt
Number of Electronics Sales: 28
```

```txt
+-------------------+-----------+----------+-----------+------------+--------+
|               date|customer_id|product_id|   category|sales_amount|quantity|
+-------------------+-----------+----------+-----------+------------+--------+
|2023-01-02 00:00:00|         93|        41|Electronics|      453.94|       5|
|2023-01-04 00:00:00|         72|        15|Electronics|      184.17|       7|
|2023-01-09 00:00:00|         75|         9|Electronics|      746.73|       2|
|2023-01-11 00:00:00|         88|         1|Electronics|      314.98|       9|
|2023-01-12 00:00:00|         24|        44|Electronics|      547.11|       8|
+-------------------+-----------+----------+-----------+------------+--------+
only showing top 5 rows
```

|      | date                | customer_id | product_id | category    | sales_amount | quantity |
| ---: | :------------------ | ----------: | ---------: | :---------- | -----------: | -------: |
|    0 | 2023-01-02 00:00:00 |          93 |         41 | Electronics |       453.94 |        5 |
|    1 | 2023-01-04 00:00:00 |          72 |         15 | Electronics |       184.17 |        7 |
|    2 | 2023-01-09 00:00:00 |          75 |          9 | Electronics |       746.73 |        2 |
|    3 | 2023-01-11 00:00:00 |          88 |          1 | Electronics |       314.98 |        9 |
|    4 | 2023-01-12 00:00:00 |          24 |         44 | Electronics |       547.11 |        8 |

</div>

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

### Pandas

When it comes to numerical filtering in Pandas, the process is similar to the previous example, where we use boolean indexing to filter rows based on a given condition condition, but here we use a numerical value instead of a string value. In the below example, we filter for sales amounts greater than `500`.

```python {.pandas linenums="1" title="Filter for high value transactions"}
high_value_sales_pd: pd.DataFrame = df_sales_pd[df_sales_pd["sales_amount"] > 500]
print(f"Number of high-value Sales: {len(high_value_sales_pd)}")
print(high_value_sales_pd.head(5))
print(high_value_sales_pd.head(5).to_markdown())
```

<div class="result" markdown>

```txt
Number of high-value Sales: 43
```

```txt
         date  customer_id  product_id     category  sales_amount  quantity
2  2023-01-03           15          29         Home        994.51         5
8  2023-01-09           75           9  Electronics        746.73         2
9  2023-01-10           75          24        Books        723.73         6
11 2023-01-12           24          44  Electronics        547.11         8
12 2023-01-13            3           8     Clothing        513.73         5
```

|      | date                | customer_id | product_id | category    | sales_amount | quantity |
| ---: | :------------------ | ----------: | ---------: | :---------- | -----------: | -------: |
|    2 | 2023-01-03 00:00:00 |          15 |         29 | Home        |       994.51 |        5 |
|    8 | 2023-01-09 00:00:00 |          75 |          9 | Electronics |       746.73 |        2 |
|    9 | 2023-01-10 00:00:00 |          75 |         24 | Books       |       723.73 |        6 |
|   11 | 2023-01-12 00:00:00 |          24 |         44 | Electronics |       547.11 |        8 |
|   12 | 2023-01-13 00:00:00 |           3 |          8 | Clothing    |       513.73 |        5 |

</div>

### SQL

When it comes to numerical filtering in SQL, the process is similar to the previous example, where we use the `WHERE` clause to filter rows based on a given condition, but here we use a numerical value instead of a string value. In the below example, we filter for sales amounts greater than `500`.

```python {.sql linenums="1" title="Filter for high value transactions"}
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

### PySpark

When it comes to numerical filtering in PySpark, the process is similar to the previous example, where we use the [`.filter()`][pyspark-filter] (or [`.where()`][pyspark-where]) method to filter rows based on a given condition, but here we use a numerical value instead of a string value. In the below example, we filter for sales amounts greater than `500`.

Also note here that we have parsed a string value to the [`.filter()`][pyspark-filter] method, instead of using the pure-Python syntax as shown above. This is because the [`.filter()`][pyspark-filter] method can accept a SQL-like string expression. This is a common practice in PySpark to parse a SQL-like string to a PySpark method.

```python {.pyspark linenums="1" title="Filter for high value transactions"}
high_value_sales_ps: psDataFrame = df_sales_ps.filter("sales_amount > 500")
print(f"Number of high-value Sales: {high_value_sales_ps.count()}")
high_value_sales_ps.show(5)
print(high_value_sales_ps.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt
Number of high-value Sales: 43
```

```txt
+-------------------+-----------+----------+-----------+------------+--------+
|               date|customer_id|product_id|   category|sales_amount|quantity|
+-------------------+-----------+----------+-----------+------------+--------+
|2023-01-03 00:00:00|         15|        29|       Home|      994.51|       5|
|2023-01-09 00:00:00|         75|         9|Electronics|      746.73|       2|
|2023-01-10 00:00:00|         75|        24|      Books|      723.73|       6|
|2023-01-12 00:00:00|         24|        44|Electronics|      547.11|       8|
|2023-01-13 00:00:00|          3|         8|   Clothing|      513.73|       5|
+-------------------+-----------+----------+-----------+------------+--------+
only showing top 5 rows
```

|      | date                | customer_id | product_id | category    | sales_amount | quantity |
| ---: | :------------------ | ----------: | ---------: | :---------- | -----------: | -------: |
|    0 | 2023-01-03 00:00:00 |          15 |         29 | Home        |       994.51 |        5 |
|    1 | 2023-01-09 00:00:00 |          75 |          9 | Electronics |       746.73 |        2 |
|    2 | 2023-01-10 00:00:00 |          75 |         24 | Books       |       723.73 |        6 |
|    3 | 2023-01-12 00:00:00 |          24 |         44 | Electronics |       547.11 |        8 |
|    4 | 2023-01-13 00:00:00 |           3 |          8 | Clothing    |       513.73 |        5 |

</div>

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


### Pandas

To select specific columns in Pandas, we can use the double square brackets syntax to specify the columns we want to keep in the DataFrame. This allows us to create a new DataFrame with only the relevant columns.

For more information about selecting specific columns, see the [Pandas documentation on selecting columns][pandas-subsetting].

```python {.pandas linenums="1" title="Select specific columns"}
sales_summary_pd: pd.DataFrame = df_sales_pd[["date", "category", "sales_amount"]]
print(f"Sales Summary DataFrame: {len(sales_summary_pd)}")
print(sales_summary_pd.head(5))
print(sales_summary_pd.head(5).to_markdown())
```

<div class="result" markdown>

```txt
Sales Summary DataFrame: 100
```

```txt
        date     category  sales_amount
0 2023-01-01         Food        490.76
1 2023-01-02  Electronics        453.94
2 2023-01-03         Home        994.51
3 2023-01-04  Electronics        184.17
4 2023-01-05         Food         27.89
```

|      | date                | category    | sales_amount |
| ---: | :------------------ | :---------- | -----------: |
|    0 | 2023-01-01 00:00:00 | Food        |       490.76 |
|    1 | 2023-01-02 00:00:00 | Electronics |       453.94 |
|    2 | 2023-01-03 00:00:00 | Home        |       994.51 |
|    3 | 2023-01-04 00:00:00 | Electronics |       184.17 |
|    4 | 2023-01-05 00:00:00 | Food        |        27.89 |

</div>

### SQL

To select specific columns in SQL, we can use the `SELECT` statement to specify the columns we want to retrieve from the table. This allows us to create a new DataFrame with only the relevant columns. We can use the [`pd.read_sql()`][pandas-read_sql] function to execute SQL queries and retrieve the data from the database.

For more information about selecting specific columns in SQL, see the [SQL SELECT statement documentation][sqlite-select].

```python {.sql linenums="1" title="Select specific columns"}
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

### PySpark

To select specific columns in PySpark, we can use the [`.select()`][pyspark-select] method to specify the columns we want to keep in the DataFrame. This allows us to create a new DataFrame with only the relevant columns. The syntax is similar to SQL, where we can specify the column names as strings.

```python {.pyspark linenums="1" title="Select specific columns"}
sales_summary_ps: psDataFrame = df_sales_ps.select("date", "category", "sales_amount")
print(f"Sales Summary DataFrame: {sales_summary_ps.count()}")
sales_summary_ps.show(5)
print(sales_summary_ps.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt
Sales Summary DataFrame: 100
```

```txt
+-------------------+-----------+------------+
|               date|   category|sales_amount|
+-------------------+-----------+------------+
|2023-01-01 00:00:00|       Food|      490.76|
|2023-01-02 00:00:00|Electronics|      453.94|
|2023-01-03 00:00:00|       Home|      994.51|
|2023-01-04 00:00:00|Electronics|      184.17|
|2023-01-05 00:00:00|       Food|       27.89|
+-------------------+-----------+------------+
only showing top 5 rows
```

|      | date                | category    | sales_amount |
| ---: | :------------------ | :---------- | -----------: |
|    0 | 2023-01-01 00:00:00 | Food        |       490.76 |
|    1 | 2023-01-02 00:00:00 | Electronics |       453.94 |
|    2 | 2023-01-03 00:00:00 | Home        |       994.51 |
|    3 | 2023-01-04 00:00:00 | Electronics |       184.17 |
|    4 | 2023-01-05 00:00:00 | Food        |        27.89 |

</div>

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

### Pandas

In Pandas, we can use the [`.agg()`][pandas-agg] method to perform aggregation operations on DataFrames. This method allows us to apply multiple aggregation functions to different columns in a single operation.

```python {.pandas linenums="1" title="Basic aggregation"}
sales_stats_pd: pd.DataFrame = df_sales_pd.agg(
    {
        "sales_amount": ["sum", "mean", "min", "max", "count"],
        "quantity": ["sum", "mean", "min", "max"],
    }
)
print(f"Sales Statistics: {len(sales_stats_pd)}")
print(sales_stats_pd)
print(sales_stats_pd.to_markdown())
```

<div class="result" markdown>

```txt
Sales Statistics: 5
```

```txt
       sales_amount  quantity
sum      48227.0500    464.00
mean       482.2705      4.64
min         15.1300      1.00
max        994.6100      9.00
count      100.0000       NaN
```

|       | sales_amount | quantity |
| :---- | -----------: | -------: |
| sum   |      48227.1 |      464 |
| mean  |      482.271 |     4.64 |
| min   |        15.13 |        1 |
| max   |       994.61 |        9 |
| count |          100 |      nan |

</div>

### SQL

In SQL, we can use the aggregate functions like `SUM()`, `AVG()`, `MIN()`, `MAX()`, and `COUNT()` to perform aggregation operations on tables.

Note here that we are _not_ using the `GROUP BY` clause, which is typically used to group rows that have the same values in specified columns into summary rows. Instead, we are performing a basic aggregation on the entire table.

```python {.sql linenums="1" title="Basic aggregation"}
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

### PySpark

In PySpark, we can use the [`.agg()`][pyspark-agg] method to perform aggregation operations on DataFrames. This method allows us to apply multiple aggregation functions to different columns in a single operation.

```python {.pyspark linenums="1" title="Basic aggregation"}
sales_stats_ps: psDataFrame = df_sales_ps.agg(
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
print(f"Sales Statistics: {sales_stats_ps.count()}")
sales_stats_ps.show(5)
print(sales_stats_ps.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt
Sales Statistics: 1
```

```txt
+---------+----------+---------+---------+-----------+------------+-------------+------------+------------+
|sales_sum|sales_mean|sales_min|sales_max|sales_count|quantity_sum|quantity_mean|quantity_min|quantity_max|
+---------+----------+---------+---------+-----------+------------+-------------+------------+------------+
| 48227.05|  482.2705|    15.13|   994.61|        100|         464|         4.64|           1|           9|
+---------+----------+---------+---------+-----------+------------+-------------+------------+------------+
```

|      | sales_sum | sales_mean | sales_min | sales_max | sales_count | quantity_sum | quantity_mean | quantity_min | quantity_max |
| ---: | --------: | ---------: | --------: | --------: | ----------: | -----------: | ------------: | -----------: | -----------: |
|    0 |   48227.1 |    482.271 |     15.13 |    994.61 |         100 |          464 |          4.64 |            1 |            9 |

</div>

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

### Pandas

This is done using the [`.groupby()`][pandas-groupby] method to group data by one or more columns and then apply aggregation functions to summarize the data, followed by the [`.agg()`][pandas-groupby-agg] method.

```python {.pandas linenums="1" title="Group by category and aggregate"}
category_sales_pd: pd.DataFrame = df_sales_pd.groupby("category").agg(
    {
        "sales_amount": ["sum", "mean", "count"],
        "quantity": "sum",
    }
)
print(f"Category Sales Summary: {len(category_sales_pd)}")
print(category_sales_pd)
print(category_sales_pd.to_markdown())
```

<div class="result" markdown>

```txt
Category Sales Summary: 5
```

```txt
            sales_amount                   quantity
                     sum        mean count      sum
category
Books           10154.83  441.514348    23      100
Clothing         7325.31  457.831875    16       62
Electronics     11407.45  407.408929    28      147
Food            12995.57  541.482083    24      115
Home             6343.89  704.876667     9       40
```

| category    | ('sales_amount', 'sum') | ('sales_amount', 'mean') | ('sales_amount', 'count') | ('quantity', 'sum') |
| :---------- | ----------------------: | -----------------------: | ------------------------: | ------------------: |
| Books       |                 10154.8 |                  441.514 |                        23 |                 100 |
| Clothing    |                 7325.31 |                  457.832 |                        16 |                  62 |
| Electronics |                 11407.5 |                  407.409 |                        28 |                 147 |
| Food        |                 12995.6 |                  541.482 |                        24 |                 115 |
| Home        |                 6343.89 |                  704.877 |                         9 |                  40 |

</div>

### SQL

In SQL, we can use the `GROUP BY` clause to group rows that have the same values in specified columns into summary rows. We can then apply aggregate functions like `SUM()`, `AVG()`, and `COUNT()` in the `SELECT` clause to summarize the data by group.

```python {.sql linenums="1" title="Group by category and aggregate"}
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

### PySpark

In PySpark, we can use the [`.groupBy()`][pyspark-groupby] method to group data by one or more columns and then apply aggregation functions using the [`.agg()`][pyspark-groupby-agg] method.

```python {.pyspark linenums="1" title="Group by category and aggregate"}
category_sales_ps: psDataFrame = df_sales_ps.groupBy("category").agg(
    F.sum("sales_amount").alias("total_sales"),
    F.avg("sales_amount").alias("average_sales"),
    F.count("*").alias("transaction_count"),
    F.sum("quantity").alias("total_quantity"),
)
print(f"Category Sales Summary: {category_sales_ps.count()}")
category_sales_ps.show(5)
print(category_sales_ps.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt
Category Sales Summary: 5
```

```txt
+-----------+------------------+------------------+-----------------+--------------+
|   category|       total_sales|     average_sales|transaction_count|total_quantity|
+-----------+------------------+------------------+-----------------+--------------+
|       Home| 6343.889999999999| 704.8766666666666|                9|            40|
|       Food|          12995.57| 541.4820833333333|               24|           115|
|Electronics|11407.449999999999|407.40892857142853|               28|           147|
|   Clothing|7325.3099999999995|457.83187499999997|               16|            62|
|      Books|          10154.83|  441.514347826087|               23|           100|
+-----------+------------------+------------------+-----------------+--------------+
```

|      | category    | total_sales | average_sales | transaction_count | total_quantity |
| ---: | :---------- | ----------: | ------------: | ----------------: | -------------: |
|    0 | Home        |     6343.89 |       704.877 |                 9 |             40 |
|    1 | Food        |     12995.6 |       541.482 |                24 |            115 |
|    2 | Electronics |     11407.4 |       407.409 |                28 |            147 |
|    3 | Clothing    |     7325.31 |       457.832 |                16 |             62 |
|    4 | Books       |     10154.8 |       441.514 |                23 |            100 |

</div>

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

### Pandas

In Pandas, we use the [`.columns`][pandas-columns] attribute of the DataFrame. This makes it easier to understand the results of the aggregation.

It's also possible to rename columns using the [`.rename()`][pandas-rename] method, which allows for more flexibility in renaming specific columns from within 'dot-method' chains.

```python {.pandas linenums="1" title="Rename columns for clarity"}
category_sales_renamed_pd: pd.DataFrame = category_sales_pd.copy()
category_sales_renamed_pd.columns = [
    "Total Sales",
    "Average Sales",
    "Transaction Count",
    "Total Quantity",
]
print(f"Renamed Category Sales Summary: {len(category_sales_renamed_pd)}")
print(category_sales_renamed_pd.head(5))
print(category_sales_renamed_pd.head(5).to_markdown())
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

### SQL

In SQL, we can use the `AS` keyword to rename columns in the `SELECT` clause. This allows us to provide more descriptive names for the aggregated columns.

In this example, we provide the same aggregation as before, but from within a subquery. Then, in the parent query, we rename the columns for clarity.

```python {.sql linenums="1" title="Rename columns for clarity"}
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

### PySpark

In PySpark, we can use the [`.withColumnsRenamed()`][pyspark-withcolumnsrenamed] method to rename columns in a DataFrame. This allows us to provide more descriptive names for the aggregated columns.

```python {.pyspark linenums="1" title="Rename columns for clarity"}
category_sales_renamed_ps: psDataFrame = category_sales_ps.withColumnsRenamed(
    {
        "total_sales": "Total Sales",
        "average_sales": "Average Sales",
        "transaction_count": "Transaction Count",
        "total_quantity": "Total Quantity",
    }
)
print(f"Renamed Category Sales Summary: {category_sales_renamed_ps.count()}")
category_sales_renamed_ps.show(5)
print(category_sales_renamed_ps.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt
Renamed Category Sales Summary: 5
```

```txt
+-----------+------------------+------------------+-----------------+--------------+
|   category|       Total Sales|     Average Sales|Transaction Count|Total Quantity|
+-----------+------------------+------------------+-----------------+--------------+
|       Home| 6343.889999999999| 704.8766666666666|                9|            40|
|       Food|          12995.57| 541.4820833333333|               24|           115|
|Electronics|11407.449999999999|407.40892857142853|               28|           147|
|   Clothing|7325.3099999999995|457.83187499999997|               16|            62|
|      Books|          10154.83|  441.514347826087|               23|           100|
+-----------+------------------+------------------+-----------------+--------------+
```

|      | category    | Total Sales | Average Sales | Transaction Count | Total Quantity |
| ---: | :---------- | ----------: | ------------: | ----------------: | -------------: |
|    0 | Home        |     6343.89 |       704.877 |                 9 |             40 |
|    1 | Food        |     12995.6 |       541.482 |                24 |            115 |
|    2 | Electronics |     11407.4 |       407.409 |                28 |            147 |
|    3 | Clothing    |     7325.31 |       457.832 |                16 |             62 |
|    4 | Books       |     10154.8 |       441.514 |                23 |            100 |

</div>

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

### Pandas

The Plotly [`px.bar()`][plotly-bar] function is able to receive a Pandas DataFrame directly, making it easy to create visualizations from the aggregated data. However, what we first need to do is to convert the index labels in to a column, so that we can use it as the x-axis in the bar chart. We do this with the [`.reset_index()`][pandas-reset_index] method.

```python {.pandas linenums="1" title="Plot the results"}
fig: go.Figure = px.bar(
    data_frame=category_sales_renamed_pd.reset_index(),
    x="category",
    y="Total Sales",
    title="Total Sales by Category",
    text="Transaction Count",
    labels={"Total Sales": "Total Sales ($)", "category": "Product Category"},
)
fig.write_html("images/pt2_total_sales_by_category_pd.html", include_plotlyjs="cdn", full_html=True)
fig.show()
```

<div class="result" markdown>


</div>

### SQL

The Plotly [`px.bar()`][plotly-bar] function can also receive a Pandas DataFrame, so we can use the results of the SQL query directly. Since the method we are using already returns the group labels in an individual column, we can use that directly in Plotly as the labels for the x-axis.

```python {.sql linenums="1" title="Plot the results"}
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


</div>

### PySpark

Plotly is unfortunately not able to directly receive a PySpark DataFrame, so we need to convert it to a Pandas DataFrame first. This is done using the [`.toPandas()`][pyspark-topandas] method, which converts the PySpark DataFrame to a Pandas DataFrame.

```python {.pyspark linenums="1" title="Plot the results"}
fig: go.Figure = px.bar(
    data_frame=category_sales_renamed_ps.toPandas(),
    x="category",
    y="Total Sales",
    title="Total Sales by Category",
    text="Transaction Count",
    labels={"Total Sales": "Total Sales ($)", "category": "Product Category"},
)
fig.write_html("images/pt2_total_sales_by_category_ps.html", include_plotlyjs="cdn", full_html=True)
fig.show()
```

<div class="result" markdown>


</div>

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

### Pandas

In Pandas, we can use the [`pd.merge()`][pandas-merge] method to combine rows from two or more tables based on a related column between them. In this case, we will join the `sales` table with the `product` table on the `product_id` column.

```python {.pandas linenums="1" title="Join sales with product data"}
sales_with_product_pd: pd.DataFrame = pd.merge(
    left=df_sales_pd,
    right=df_product_pd[["product_id", "product_name", "price"]],
    on="product_id",
    how="left",
)
print(f"Sales with Product Information: {len(sales_with_product_pd)}")
print(sales_with_product_pd.head(5))
print(sales_with_product_pd.head(5).to_markdown())
```

<div class="result" markdown>

```txt
Sales with Product Information: 100
```

```txt
        date  customer_id  product_id     category  sales_amount  quantity  product_name   price
0 2023-01-01           52          45         Food        490.76         7    Product 45  493.14
1 2023-01-02           93          41  Electronics        453.94         5    Product 41  193.39
2 2023-01-03           15          29         Home        994.51         5    Product 29   80.07
3 2023-01-04           72          15  Electronics        184.17         7    Product 15  153.67
4 2023-01-05           61          45         Food         27.89         9    Product 45  493.14
```

|      | date                | customer_id | product_id | category    | sales_amount | quantity | product_name |  price |
| ---: | :------------------ | ----------: | ---------: | :---------- | -----------: | -------: | :----------- | -----: |
|    0 | 2023-01-01 00:00:00 |          52 |         45 | Food        |       490.76 |        7 | Product 45   | 493.14 |
|    1 | 2023-01-02 00:00:00 |          93 |         41 | Electronics |       453.94 |        5 | Product 41   | 193.39 |
|    2 | 2023-01-03 00:00:00 |          15 |         29 | Home        |       994.51 |        5 | Product 29   |  80.07 |
|    3 | 2023-01-04 00:00:00 |          72 |         15 | Electronics |       184.17 |        7 | Product 15   | 153.67 |
|    4 | 2023-01-05 00:00:00 |          61 |         45 | Food        |        27.89 |        9 | Product 45   | 493.14 |

</div>

### SQL

In SQL, we can use the [`JOIN`][sqlite-tutorial-join] clause to combine rows from two or more tables based on a related column between them. In this case, we will join the `sales` table with the `product` table on the `product_id` column.

```python {.sql linenums="1" title="Join sales with product data"}
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

### PySpark

In PySpark, we can use the [`.join()`][pyspark-join] method to combine rows from two or more DataFrames based on a related column between them. In this case, we will join the `sales` DataFrame with the `product` DataFrame on the `product_id` column.

```python {.pyspark linenums="1" title="Join sales with product data"}
sales_with_product_ps: psDataFrame = df_sales_ps.join(
    other=df_product_ps.select("product_id", "product_name", "price"),
    on="product_id",
    how="left",
)
print(f"Sales with Product Information: {sales_with_product_ps.count()}")
sales_with_product_ps.show(5)
print(sales_with_product_ps.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt
Sales with Product Information: 100
```

```txt
+----------+-------------------+-----------+-----------+------------+--------+------------+------+
|product_id|               date|customer_id|   category|sales_amount|quantity|product_name| price|
+----------+-------------------+-----------+-----------+------------+--------+------------+------+
|         1|2023-01-06 00:00:00|         21|   Clothing|      498.95|       5|   Product 1|257.57|
|         1|2023-01-11 00:00:00|         88|Electronics|      314.98|       9|   Product 1|257.57|
|         1|2023-02-11 00:00:00|         55|       Food|       199.0|       5|   Product 1|257.57|
|         1|2023-04-04 00:00:00|         85|       Food|      146.97|       7|   Product 1|257.57|
|         5|2023-01-21 00:00:00|         64|Electronics|      356.58|       5|   Product 5|200.71|
+----------+-------------------+-----------+-----------+------------+--------+------------+------+
only showing top 5 rows
```

|      | product_id | date                | customer_id | category    | sales_amount | quantity | product_name |  price |
| ---: | ---------: | :------------------ | ----------: | :---------- | -----------: | -------: | :----------- | -----: |
|    0 |          1 | 2023-01-11 00:00:00 |          88 | Electronics |       314.98 |        9 | Product 1    | 257.57 |
|    1 |          1 | 2023-02-11 00:00:00 |          55 | Food        |          199 |        5 | Product 1    | 257.57 |
|    2 |          5 | 2023-01-21 00:00:00 |          64 | Electronics |       356.58 |        5 | Product 5    | 200.71 |
|    3 |          5 | 2023-02-18 00:00:00 |          39 | Books       |        79.71 |        8 | Product 5    | 200.71 |
|    4 |          6 | 2023-03-23 00:00:00 |          34 | Electronics |        48.45 |        8 | Product 6    |  15.31 |

</div>

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

### Pandas

This process is similar to the previous step, but now we will extend the `sales_with_product` DataFrame to join it with the `customer` DataFrame on the `customer_id` column. This will give us a complete view of the sales data, including product and customer details.

```python {.pandas linenums="1" title="Join with customer information to get a complete view"}
complete_sales_pd: pd.DataFrame = pd.merge(
    sales_with_product_pd,
    df_customer_pd[["customer_id", "customer_name", "city", "state"]],
    on="customer_id",
    how="left",
)
print(f"Complete Sales Data with Customer Information: {len(complete_sales_pd)}")
print(complete_sales_pd.head(5))
print(complete_sales_pd.head(5).to_markdown())
```

<div class="result" markdown>

```txt
Complete Sales Data with Customer Information: 100
```

```txt
        date  customer_id  product_id     category  sales_amount  quantity  product_name   price  customer_name      city  state
0 2023-01-01           52          45         Food        490.76         7    Product 45  493.14    Customer 52   Phoenix     TX
1 2023-01-02           93          41  Electronics        453.94         5    Product 41  193.39    Customer 93  New York     TX
2 2023-01-03           15          29         Home        994.51         5    Product 29   80.07    Customer 15  New York     CA
3 2023-01-04           72          15  Electronics        184.17         7    Product 15  153.67    Customer 72   Houston     IL
4 2023-01-05           61          45         Food         27.89         9    Product 45  493.14    Customer 61   Phoenix     IL
```

|      | date                | customer_id | product_id | category    | sales_amount | quantity | product_name |  price | customer_name | city     | state |
| ---: | :------------------ | ----------: | ---------: | :---------- | -----------: | -------: | :----------- | -----: | :------------ | :------- | :---- |
|    0 | 2023-01-01 00:00:00 |          52 |         45 | Food        |       490.76 |        7 | Product 45   | 493.14 | Customer 52   | Phoenix  | TX    |
|    1 | 2023-01-02 00:00:00 |          93 |         41 | Electronics |       453.94 |        5 | Product 41   | 193.39 | Customer 93   | New York | TX    |
|    2 | 2023-01-03 00:00:00 |          15 |         29 | Home        |       994.51 |        5 | Product 29   |  80.07 | Customer 15   | New York | CA    |
|    3 | 2023-01-04 00:00:00 |          72 |         15 | Electronics |       184.17 |        7 | Product 15   | 153.67 | Customer 72   | Houston  | IL    |
|    4 | 2023-01-05 00:00:00 |          61 |         45 | Food        |        27.89 |        9 | Product 45   | 493.14 | Customer 61   | Phoenix  | IL    |

</div>

### SQL

This process is similar to the previous step, but now we will extend the `sales_with_product` DataFrame to join it with the `customer` DataFrame on the `customer_id` column. This will give us a complete view of the sales data, including product and customer details.

```python {.sql linenums="1" title="Join with customer information to get a complete view"}
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

### PySpark

This process is similar to the previous step, but now we will extend the `sales_with_product` DataFrame to join it with the `customer` DataFrame on the `customer_id` column. This will give us a complete view of the sales data, including product and customer details.

```python {.pyspark linenums="1" title="Join with customer information to get a complete view"}
complete_sales_ps: psDataFrame = sales_with_product_ps.alias("s").join(
    other=df_customer_ps.select("customer_id", "customer_name", "city", "state").alias("c"),
    on="customer_id",
    how="left",
)
print(f"Complete Sales Data with Customer Information: {complete_sales_ps.count()}")
complete_sales_ps.show(5)
print(complete_sales_ps.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt
Complete Sales Data with Customer Information: 100
```

```txt
+-----------+----------+-------------------+-----------+------------+--------+------------+------+-------------+-----------+-----+
|customer_id|product_id|               date|   category|sales_amount|quantity|product_name| price|customer_name|       city|state|
+-----------+----------+-------------------+-----------+------------+--------+------------+------+-------------+-----------+-----+
|         39|         5|2023-02-18 00:00:00|      Books|       79.71|       8|   Product 5|200.71|  Customer 39|Los Angeles|   NY|
|         88|         1|2023-01-11 00:00:00|Electronics|      314.98|       9|   Product 1|257.57|  Customer 88|Los Angeles|   TX|
|         85|         1|2023-04-04 00:00:00|       Food|      146.97|       7|   Product 1|257.57|  Customer 85|    Phoenix|   CA|
|         55|         1|2023-02-11 00:00:00|       Food|       199.0|       5|   Product 1|257.57|  Customer 55|Los Angeles|   NY|
|         21|         1|2023-01-06 00:00:00|   Clothing|      498.95|       5|   Product 1|257.57|  Customer 21|Los Angeles|   IL|
+-----------+----------+-------------------+-----------+------------+--------+------------+------+-------------+-----------+-----+
only showing top 5 rows
```

|      | customer_id | product_id | date                | category    | sales_amount | quantity | product_name |  price | customer_name | city        | state |
| ---: | ----------: | ---------: | :------------------ | :---------- | -----------: | -------: | :----------- | -----: | :------------ | :---------- | :---- |
|    0 |          88 |          1 | 2023-01-11 00:00:00 | Electronics |       314.98 |        9 | Product 1    | 257.57 | Customer 88   | Los Angeles | TX    |
|    1 |          55 |          1 | 2023-02-11 00:00:00 | Food        |          199 |        5 | Product 1    | 257.57 | Customer 55   | Los Angeles | NY    |
|    2 |          64 |          5 | 2023-01-21 00:00:00 | Electronics |       356.58 |        5 | Product 5    | 200.71 | Customer 64   | Los Angeles | NY    |
|    3 |          39 |          5 | 2023-02-18 00:00:00 | Books       |        79.71 |        8 | Product 5    | 200.71 | Customer 39   | Los Angeles | NY    |
|    4 |          34 |          6 | 2023-03-23 00:00:00 | Electronics |        48.45 |        8 | Product 6    |  15.31 | Customer 34   | Los Angeles | NY    |

</div>

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

### Pandas

In Pandas, we can calculate the revenue for each sale by multiplying the `price` and `quantity` columns. We can then compare this calculated revenue with the `sales_amount` column to identify any discrepancies.

Notice here that the syntax for Pandas uses the `DataFrame` object directly, and we can access the columns using the 'slice' (`[]`) operator.

```python {.pandas linenums="1" title="Calculate revenue and compare with sales amount"}
complete_sales_pd["calculated_revenue"] = complete_sales_pd["price"] * complete_sales_pd["quantity"]
complete_sales_pd["price_difference"] = complete_sales_pd["sales_amount"] - complete_sales_pd["calculated_revenue"]
print(f"Complete Sales Data with Calculated Revenue and Price Difference: {len(complete_sales_pd)}")
print(complete_sales_pd[["sales_amount", "price", "quantity", "calculated_revenue", "price_difference"]].head(5))
print(
    complete_sales_pd[["sales_amount", "price", "quantity", "calculated_revenue", "price_difference"]]
    .head(5)
    .to_markdown()
)
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

### SQL

In SQL, we can calculate the revenue for each sale by multiplying the `price` and `quantity` columns. We can then compare this calculated revenue with the `sales_amount` column to identify any discrepancies.

```python {.sql linenums="1" title="Calculate revenue and compare with sales amount"}
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

### PySpark

In PySpark, we can calculate the revenue for each sale by multiplying the `price` and `quantity` columns. We can then compare this calculated revenue with the `sales_amount` column to identify any discrepancies.

Notice here that the syntax for PySpark uses the [`.withColumns`][pyspark-withcolumns] method to add new multiple columns to the DataFrame simultaneously. This method takes a dictionary where the keys are the names of the new columns and the values are the expressions to compute those columns. The methematical computation we have shown here uses two different methods:

1. With the PySpark API, we can use the [`F.col()`][pyspark-col] function to refer to the columns, and multiply them directly
2. With the Spark SQL API, we can use the [`F.expr()`][pyspark-expr] function to write a SQL-like expression for the calculation.

```python {.pyspark linenums="1" title="Calculate revenue and compare with sales amount"}
complete_sales_ps: psDataFrame = complete_sales_ps.withColumns(
    {
        "calculated_revenue": F.col("price") * F.col("quantity"),
        "price_difference": F.expr("sales_amount - (price * quantity)"),
    },
).select("sales_amount", "price", "quantity", "calculated_revenue", "price_difference")
print(f"Complete Sales Data with Calculated Revenue and Price Difference: {complete_sales_ps.count()}")
complete_sales_ps.show(5)
print(complete_sales_ps.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt
Complete Sales Data with Calculated Revenue and Price Difference: 100
```

```txt
+------------+------+--------+------------------+------------------+
|sales_amount| price|quantity|calculated_revenue|  price_difference|
+------------+------+--------+------------------+------------------+
|       79.71|200.71|       8|           1605.68|          -1525.97|
|      314.98|257.57|       9|           2318.13|          -2003.15|
|      146.97|257.57|       7|           1802.99|          -1656.02|
|       199.0|257.57|       5|           1287.85|          -1088.85|
|      498.95|257.57|       5|           1287.85|-788.8999999999999|
+------------+------+--------+------------------+------------------+
only showing top 5 rows
```

|      | sales_amount |  price | quantity | calculated_revenue | price_difference |
| ---: | -----------: | -----: | -------: | -----------------: | ---------------: |
|    0 |        48.45 |  15.31 |        8 |             122.48 |           -74.03 |
|    1 |        79.71 | 200.71 |        8 |            1605.68 |         -1525.97 |
|    2 |       314.98 | 257.57 |        9 |            2318.13 |         -2003.15 |
|    3 |          199 | 257.57 |        5 |            1287.85 |         -1088.85 |
|    4 |       356.58 | 200.71 |        5 |            1003.55 |          -646.97 |

</div>

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

### Pandas

In Pandas, we can use the [`.groupby()`][pandas-groupby] method to group the data by the `date` column, followed by the [`.agg()`][pandas-groupby-agg] method to calculate the total sales for each day. This will then set us up for further time-based calculations in the following steps

```python {.pandas linenums="1" title="Time-based window function"}
df_sales_pd["date"] = pd.to_datetime(df_sales_pd["date"])  # Ensure correct date type
daily_sales_pd: pd.DataFrame = (
    df_sales_pd.groupby(df_sales_pd["date"].dt.date)
    .agg(total_sales=("sales_amount", "sum"))
    .reset_index()
    .sort_values("date")
)
print(f"Daily Sales Summary: {len(daily_sales_pd)}")
print(daily_sales_pd.head(5))
print(daily_sales_pd.head(5).to_markdown())
```

<div class="result" markdown>

```txt
Daily Sales Summary: 100
```

```txt
         date  total_sales
0  2023-01-01        490.76
1  2023-01-02        453.94
2  2023-01-03        994.51
3  2023-01-04        184.17
4  2023-01-05         27.89
```

|      | date       | total_sales |
| ---: | :--------- | ----------: |
|    0 | 2023-01-01 |      490.76 |
|    1 | 2023-01-02 |      453.94 |
|    2 | 2023-01-03 |      994.51 |
|    3 | 2023-01-04 |      184.17 |
|    4 | 2023-01-05 |       27.89 |

</div>

### SQL

In SQL, we can use the `GROUP BY` clause to group the data by the `date` column and then use the `SUM()` function to calculate the total sales for each day. This will give us a daily summary of sales, which we can then use to analyze trends over time.

```python {.sql linenums="1" title="Time-based window function"}
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

### PySpark

In PySpark, we can use the [`.groupBy()`][pyspark-groupby] method to group the data by the `date` column, followed by the [`.agg()`][pyspark-groupby-agg] method to calculate the total sales for each day. This will then set us up for further time-based calculations in the following steps.

```python {.pyspark linenums="1" title="Time-based window function"}
df_sales_ps: psDataFrame = df_sales_ps.withColumn("date", F.to_date(df_sales_ps["date"]))
daily_sales_ps: psDataFrame = (
    df_sales_ps.groupBy("date").agg(F.sum("sales_amount").alias("total_sales")).orderBy("date")
)
print(f"Daily Sales Summary: {daily_sales_ps.count()}")
daily_sales_ps.show(5)
print(daily_sales_ps.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt
Daily Sales Summary: 100
```

```txt
+----------+-----------+
|      date|total_sales|
+----------+-----------+
|2023-01-01|     490.76|
|2023-01-02|     453.94|
|2023-01-03|     994.51|
|2023-01-04|     184.17|
|2023-01-05|      27.89|
+----------+-----------+
only showing top 5 rows
```

|      | date       | total_sales |
| ---: | :--------- | ----------: |
|    0 | 2023-01-01 |      490.76 |
|    1 | 2023-01-02 |      453.94 |
|    2 | 2023-01-03 |      994.51 |
|    3 | 2023-01-04 |      184.17 |
|    4 | 2023-01-05 |       27.89 |

</div>

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

### Pandas

In Pandas, we can calculate the lag and lead values for the sales amount by using the [`.shift()`][pandas-shift] method. This method shifts the values in a column by a specified number of periods, allowing us to create lag and lead columns.

Note that the [`.shift()`][pandas-shift] method simply shifts the values in the column by a number of rows up or down, so we can use it to create lag and lead columns. This function itself does not need to be ordered because it assumes that the DataFrame is already ordered. However, if you want it to be ordered, you can use the [`.sort_values()`][pandas-sort_values] method before applying [`.shift()`][pandas-shift].

```python {.pandas linenums="1" title="Calculate lag and lead"}
daily_sales_pd["previous_day_sales"] = daily_sales_pd["total_sales"].shift(1)
daily_sales_pd["next_day_sales"] = daily_sales_pd["total_sales"].shift(-1)
print(f"Daily Sales with Lag and Lead: {len(daily_sales_pd)}")
print(daily_sales_pd.head(5))
print(daily_sales_pd.head(5).to_markdown())
```

<div class="result" markdown>

```txt
Daily Sales with Lag and Lead: 100
```

```txt
         date  total_sales  previous_day_sales  next_day_sales
0  2023-01-01       490.76                 NaN          453.94
1  2023-01-02       453.94              490.76          994.51
2  2023-01-03       994.51              453.94          184.17
3  2023-01-04       184.17              994.51           27.89
4  2023-01-05        27.89              184.17          498.95
```

|      | date       | total_sales | previous_day_sales | next_day_sales |
| ---: | :--------- | ----------: | -----------------: | -------------: |
|    0 | 2023-01-01 |      490.76 |                nan |         453.94 |
|    1 | 2023-01-02 |      453.94 |             490.76 |         994.51 |
|    2 | 2023-01-03 |      994.51 |             453.94 |         184.17 |
|    3 | 2023-01-04 |      184.17 |             994.51 |          27.89 |
|    4 | 2023-01-05 |       27.89 |             184.17 |         498.95 |

</div>

### SQL

In SQL, we can use the `LAG()` and `LEAD()` window functions to calculate the lag and lead values for the sales amount. These functions allow us to access data from previous and next rows in the result set without needing to join the table to itself.

The part that is important to note here is that the `LAG()` and `LEAD()` functions are used in conjunction with the `OVER` clause, which defines the window over which the function operates. In this case, we are ordering by the `date` column to ensure that the lag and lead values are calculated based on the chronological order of the sales data.

```python {.sql linenums="1" title="Calculate lag and lead"}
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

### PySpark

In PySpark, we can use the [`.lag()`][pyspark-lag] and [`.lead()`][pyspark-lead] functions to calculate the lag and lead values for the sales amount. These functions are used in conjunction with a window specification that defines the order of the rows.

Note that in PySpark, we can define a Window function in one of two ways: using the PySpark API or using the Spark SQL API.

1. **The PySpark API**: The PySpark API allows us to define a window specification using the [`Window()`][pyspark-window] class, which provides methods to specify the ordering of the rows. We can then use the `F.lag()` and `F.lead()` functions to calculate the lag and lead values _over_ a given window on the table.
2. **The Spark SQL API**: The Spark SQL API is used through the [`F.expr()`][pyspark-expr] function, which allows us to write SQL-like expressions for the calculations. This is similar to how we would write SQL queries, but it is executed within the PySpark context.

Here in the below example, we show how the previous day sales can be calculated using the [`.lag()`][pyspark-lag] function in the PySpark API, and the next day sales can be calculated using the [`LEAD()`][sparksql-lead] function in the Spark SQL API. Functionally, both of these two methods achieve the same result, but aesthetically they use slightly different syntax. It is primarily a matter of preference which one you choose to use.

```python {.pyspark linenums="1" title="Calculate lag and lead"}
window_spec_ps: Window = Window.orderBy("date")
daily_sales_ps: psDataFrame = daily_sales_ps.withColumns(
    {
        "previous_day_sales": F.lag("total_sales").over(window_spec_ps),
        "next_day_sales": F.expr("LEAD(total_sales) OVER (ORDER BY date)"),
    },
)
print(f"Daily Sales with Lag and Lead: {daily_sales_ps.count()}")
daily_sales_ps.show(5)
print(daily_sales_ps.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt
Daily Sales with Lag and Lead: 100
```

```txt
+----------+-----------+------------------+--------------+
|      date|total_sales|previous_day_sales|next_day_sales|
+----------+-----------+------------------+--------------+
|2023-01-01|     490.76|              NULL|        453.94|
|2023-01-02|     453.94|            490.76|        994.51|
|2023-01-03|     994.51|            453.94|        184.17|
|2023-01-04|     184.17|            994.51|         27.89|
|2023-01-05|      27.89|            184.17|        498.95|
+----------+-----------+------------------+--------------+
only showing top 5 rows
```

|      | date       | total_sales | previous_day_sales | next_day_sales |
| ---: | :--------- | ----------: | -----------------: | -------------: |
|    0 | 2023-01-01 |      490.76 |                nan |         453.94 |
|    1 | 2023-01-02 |      453.94 |             490.76 |         994.51 |
|    2 | 2023-01-03 |      994.51 |             453.94 |         184.17 |
|    3 | 2023-01-04 |      184.17 |             994.51 |          27.89 |
|    4 | 2023-01-05 |       27.89 |             184.17 |         498.95 |

</div>

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

### Pandas

In Pandas, we can calculate the day-over-day change in sales by subtracting the `previous_day_sales` column from the `total_sales` column. This is a fairly straight-forward calculation.

We can also calculate the percentage change in sales using the [`.pct_change()`][pandas-pct_change] method, which calculates the percentage change between the current and previous values. Under the hood, this method calculates the fractional change using the formula:

```txt
((value_current_row - value_previous_row) / value_previous_row)
```

So therefore we need to multiple the result by `100`.

```python {.pandas linenums="1" title="Calculate day-over-day change"}
daily_sales_pd["day_over_day_change"] = daily_sales_pd["total_sales"] - daily_sales_pd["previous_day_sales"]
daily_sales_pd["pct_change"] = daily_sales_pd["total_sales"].pct_change() * 100
print(f"Daily Sales with Day-over-Day Change: {len(daily_sales_pd)}")
print(daily_sales_pd.head(5))
print(daily_sales_pd.head(5).to_markdown())
```

<div class="result" markdown>

```txt
Daily Sales with Day-over-Day Change: 100
```

```txt
         date  total_sales  previous_day_sales  next_day_sales  day_over_day_change  day_over_day_change  7d_moving_avg
0  2023-01-01       490.76                 NaN          453.94                  NaN                  NaN     490.760000
1  2023-01-02       453.94              490.76          994.51               -36.82               -36.82     472.350000
2  2023-01-03       994.51              453.94          184.17               540.57               540.57     646.403333
3  2023-01-04       184.17              994.51           27.89              -810.34              -810.34     530.845000
4  2023-01-05        27.89              184.17          498.95              -156.28              -156.28     430.254000
```

|      | date       | total_sales | previous_day_sales | next_day_sales | pct_change | day_over_day_change | 7d_moving_avg |
| ---: | :--------- | ----------: | -----------------: | -------------: | ---------: | ------------------: | ------------: |
|    0 | 2023-01-01 |      490.76 |                nan |         453.94 |        nan |                 nan |        490.76 |
|    1 | 2023-01-02 |      453.94 |             490.76 |         994.51 |   -7.50265 |              -36.82 |        472.35 |
|    2 | 2023-01-03 |      994.51 |             453.94 |         184.17 |    119.084 |              540.57 |       646.403 |
|    3 | 2023-01-04 |      184.17 |             994.51 |          27.89 |   -81.4813 |             -810.34 |       530.845 |
|    4 | 2023-01-05 |       27.89 |             184.17 |         498.95 |   -84.8564 |             -156.28 |       430.254 |

</div>

### SQL

In SQL, we can calculate the day-over-day change in sales by subtracting the `previous_day_sales` column from the `total_sales` column. We can also calculate the percentage change in sales using the formula:

```txt
((current_day_sales - previous_day_sales) / previous_day_sales) * 100
```

```python {.sql linenums="1" title="Day-over-day change already calculated"}
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

### PySpark

In PySpark, we can calculate the day-over-day change in sales by subtracting the `previous_day_sales` column from the `total_sales` column. We can also calculate the percentage change in sales using the formula:

```txt
((current_day_sales - previous_day_sales) / previous_day_sales) * 100
```

Here, we have again shown these calculations using two different methods: using the PySpark API and using the Spark SQL API. Realistically, the results for  both of them can be achieved using either method.

```python {.pyspark linenums="1" title="Calculate day-over-day change"}
daily_sales_ps: psDataFrame = daily_sales_ps.withColumns(
    {
        "day_over_day_change": F.col("total_sales") - F.col("previous_day_sales"),
        "pct_change": F.expr("((total_sales - previous_day_sales) / previous_day_sales) * 100").alias("pct_change"),
    }
)
print(f"Daily Sales with Day-over-Day Change: {daily_sales_ps.count()}")
daily_sales_ps.show(5)
print(daily_sales_ps.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt
Daily Sales with Day-over-Day Change: 100
```

```txt
+----------+-----------+------------------+--------------+-------------------+------------------+
|      date|total_sales|previous_day_sales|next_day_sales|day_over_day_change|        pct_change|
+----------+-----------+------------------+--------------+-------------------+------------------+
|2023-01-01|     490.76|              NULL|        453.94|               NULL|              NULL|
|2023-01-02|     453.94|            490.76|        994.51| -36.81999999999999|-7.502648952644875|
|2023-01-03|     994.51|            453.94|        184.17|  540.5699999999999|119.08401991452612|
|2023-01-04|     184.17|            994.51|         27.89|            -810.34|-81.48133251551015|
|2023-01-05|      27.89|            184.17|        498.95|-156.27999999999997|-84.85638268990606|
+----------+-----------+------------------+--------------+-------------------+------------------+
only showing top 5 rows
```

|      | date       | total_sales | previous_day_sales | next_day_sales | day_over_day_change | pct_change |
| ---: | :--------- | ----------: | -----------------: | -------------: | ------------------: | ---------: |
|    0 | 2023-01-01 |      490.76 |                nan |         453.94 |                 nan |        nan |
|    1 | 2023-01-02 |      453.94 |             490.76 |         994.51 |              -36.82 |   -7.50265 |
|    2 | 2023-01-03 |      994.51 |             453.94 |         184.17 |              540.57 |    119.084 |
|    3 | 2023-01-04 |      184.17 |             994.51 |          27.89 |             -810.34 |   -81.4813 |
|    4 | 2023-01-05 |       27.89 |             184.17 |         498.95 |             -156.28 |   -84.8564 |

</div>

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

### Pandas

In Pandas, we can calculate the 7-day moving average of sales using the [`.rolling()`][pandas-rolling] method. This method allows us to specify a window size (in this case, `window=7` which is 7 days) and calculate the mean over that window. The `min_periods` parameter ensures that we get a value even if there are fewer than 7 days of data available at the start of the series. Finally, the [`.mean()`][pandas-rolling-mean] method calculates the average over the specified window.

```python {.pandas linenums="1" title="Calculate 7-day moving average"}
daily_sales_pd["7d_moving_avg"] = daily_sales_pd["total_sales"].rolling(window=7, min_periods=1).mean()
print(f"Daily Sales with 7-Day Moving Average: {len(daily_sales_pd)}")
print(daily_sales_pd.head(5))
print(daily_sales_pd.head(5).to_markdown())
```

<div class="result" markdown>

```txt
Daily Sales with 7-Day Moving Average: 100
```

```txt
         date  total_sales  previous_day_sales  next_day_sales  day_over_day_change  pct_change  7d_moving_avg
0  2023-01-01       490.76                 NaN          453.94                  NaN         NaN     490.760000
1  2023-01-02       453.94              490.76          994.51               -36.82   -7.502649     472.350000
2  2023-01-03       994.51              453.94          184.17               540.57  119.084020     646.403333
3  2023-01-04       184.17              994.51           27.89              -810.34  -81.481333     530.845000
4  2023-01-05        27.89              184.17          498.95              -156.28  -84.856383     430.254000
```

|      | date       | total_sales | previous_day_sales | next_day_sales | day_over_day_change | pct_change | 7d_moving_avg |
| ---: | :--------- | ----------: | -----------------: | -------------: | ------------------: | ---------: | ------------: |
|    0 | 2023-01-01 |      490.76 |                nan |         453.94 |                 nan |        nan |        490.76 |
|    1 | 2023-01-02 |      453.94 |             490.76 |         994.51 |              -36.82 |   -7.50265 |        472.35 |
|    2 | 2023-01-03 |      994.51 |             453.94 |         184.17 |              540.57 |    119.084 |       646.403 |
|    3 | 2023-01-04 |      184.17 |             994.51 |          27.89 |             -810.34 |   -81.4813 |       530.845 |
|    4 | 2023-01-05 |       27.89 |             184.17 |         498.95 |             -156.28 |   -84.8564 |       430.254 |

</div>

### SQL

In SQL, we can calculate the 7-day moving average of sales using the `AVG()` window function with the `OVER` clause. It is important to include this `OVER` clause, because it is what the SQL engine uses to determine that it should be a Window function, rather than a regular aggregate function (which is specified using the `GROUP BY` clause).

Here in our example, there are three different parts to the Window function:

1. **The `ORDER BY` clause**: This specifies the order of the rows in the window. In this case, we are ordering by the `sale_date` column.
2. **The `ROWS BETWEEN` clause**: This specifies the range of rows to include in the window. In this case, we are including the number of rows from 6 preceding rows to the current row. This means that for each row, the window will include the current row and the 6 rows before it, giving us a total of 7 rows in the window. It is important that you specify the `ORDER BY` clause before the `ROWS BETWEEN` clause to ensure that the correct rows are included in the window.
3. **The `AVG()` function**: This calculates the average of the `total_sales` column over the specified window.

Another perculiarity to note here is around the use of the sub-query. The sub-query is used to first calculate the daily sales, including the previous and next day sales, and the day-over-day change. This is because we need to calculate the moving average over the daily sales, rather than the individual sales transactions. The sub-query allows us to aggregate the sales data by date before calculating the moving average. The only change that we are including in the outer-query is the addition of the moving average calculation.

```python {.sql linenums="1" title="Calculate 7-day moving average"}
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

### PySpark

In PySpark, we can calculate the 7-day moving average of sales using the [`F.avg()`][pyspark-avg] function in combination with the [`Window()`][pyspark-window] class. The [`Window()`][pyspark-window] class allows us to define a window specification for the calculation. We can use the [`.orderBy()`][pyspark-window-orderby] method to specify the order of the rows in the window, and the [`.rowsBetween()`][pyspark-window-rowsbetween] method to specify the range of rows to include in the window. The [`F.avg()`][pyspark-avg] function is then able to calculate the average of the `total_sales` column over the specified window.

As with many aspects of PySpark, there are multiple ways to achieve the same result. In this case, we can use either the [`F.avg()`][pyspark-avg] function with the [`Window()`][pyspark-window] class, or we can use the SQL expression syntax with the [`F.expr()`][pyspark-expr] function. Both methods will yield the same result.

```python {.pyspark linenums="1" title="Calculate 7-day moving average"}
daily_sales_ps: psDataFrame = daily_sales_ps.withColumns(
    {
        "7d_moving_avg": F.avg("total_sales").over(Window.orderBy("date").rowsBetween(-6, 0)),
        "7d_rolling_avg": F.expr("AVG(total_sales) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)"),
    }
)
print(f"Daily Sales with 7-Day Moving Average: {daily_sales_ps.count()}")
daily_sales_ps.show(5)
print(daily_sales_ps.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt
Daily Sales with 7-Day Moving Average: 100
```

```txt
+----------+-----------+------------------+--------------+-------------------+------------------+-----------------+-----------------+
|      date|total_sales|previous_day_sales|next_day_sales|day_over_day_change|        pct_change|    7d_moving_avg|   7d_rolling_avg|
+----------+-----------+------------------+--------------+-------------------+------------------+-----------------+-----------------+
|2023-01-01|     490.76|              NULL|        453.94|               NULL|              NULL|           490.76|           490.76|
|2023-01-02|     453.94|            490.76|        994.51| -36.81999999999999|-7.502648952644875|           472.35|           472.35|
|2023-01-03|     994.51|            453.94|        184.17|  540.5699999999999|119.08401991452612|646.4033333333333|646.4033333333333|
|2023-01-04|     184.17|            994.51|         27.89|            -810.34|-81.48133251551015|          530.845|          530.845|
|2023-01-05|      27.89|            184.17|        498.95|-156.27999999999997|-84.85638268990606|          430.254|          430.254|
+----------+-----------+------------------+--------------+-------------------+------------------+-----------------+-----------------+
only showing top 5 rows
```

|      | date       | total_sales | previous_day_sales | next_day_sales | day_over_day_change | pct_change | 7d_moving_avg | 7d_rolling_avg |
| ---: | :--------- | ----------: | -----------------: | -------------: | ------------------: | ---------: | ------------: | -------------: |
|    0 | 2023-01-01 |      490.76 |                nan |         453.94 |                 nan |        nan |        490.76 |         490.76 |
|    1 | 2023-01-02 |      453.94 |             490.76 |         994.51 |              -36.82 |   -7.50265 |        472.35 |         472.35 |
|    2 | 2023-01-03 |      994.51 |             453.94 |         184.17 |              540.57 |    119.084 |       646.403 |        646.403 |
|    3 | 2023-01-04 |      184.17 |             994.51 |          27.89 |             -810.34 |   -81.4813 |       530.845 |        530.845 |
|    4 | 2023-01-05 |       27.89 |             184.17 |         498.95 |             -156.28 |   -84.8564 |       430.254 |        430.254 |

</div>

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

### Pandas

Plotly is easily able to handle Pandas DataFrames, so we can directly parse the columns from the DataFrame to create the traces for the daily sales and the 7-day moving average.

```python {.pandas linenums="1" title="Plot results"}
fig: go.Figure = (
    go.Figure()
    .add_trace(
        go.Scatter(
            x=daily_sales_pd["date"],
            y=daily_sales_pd["total_sales"],
            mode="lines",
            name="Daily Sales",
        )
    )
    .add_trace(
        go.Scatter(
            x=daily_sales_pd["date"],
            y=daily_sales_pd["7d_moving_avg"],
            mode="lines",
            name="7-Day Moving Average",
            line_width=3,
        ),
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
fig.write_html("images/pt4_daily_sales_with_7d_avg_pd.html", include_plotlyjs="cdn", full_html=True)
fig.show()
```

<div class="result" markdown>


</div>

### SQL

Plotly is easily able to handle Pandas DataFrames, so we can directly parse the columns from the DataFrame to create the traces for the daily sales and the 7-day moving average.

```python {.sql linenums="1" title="Plot results"}
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


</div>

### PySpark

Plotly is not able to interpret PySpark DataFrames directly, so we need to convert the PySpark DataFrame to a Pandas DataFrame before plotting. This can be done using the [`.toPandas()`][pyspark-topandas] method. We can then parse the columns from the Pandas DataFrame to create the traces for the daily sales and the 7-day moving average.

```python {.pyspark linenums="1" title="Plot results"}
fig: go.Figure = (
    go.Figure()
    .add_trace(
        go.Scatter(
            x=daily_sales_ps.toPandas()["date"],
            y=daily_sales_ps.toPandas()["total_sales"],
            mode="lines",
            name="Daily Sales",
        )
    )
    .add_trace(
        go.Scatter(
            x=daily_sales_ps.toPandas()["date"],
            y=daily_sales_ps.toPandas()["7d_moving_avg"],
            mode="lines",
            name="7-Day Moving Average",
            line_width=3,
        ),
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
fig.write_html("images/pt4_daily_sales_with_7d_avg_ps.html", include_plotlyjs="cdn", full_html=True)
fig.show()
```

<div class="result" markdown>


</div>

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

### Pandas

In Pandas, we can use the [`.rank()`][pandas-rank] method to rank values in a DataFrame. This method allows us to specify the ranking method (e.g., dense, average, min, max) and whether to rank in ascending or descending order.

```python {.pandas linenums="1" title="Rank customers by total spending"}
customer_spending_pd: pd.DataFrame = df_sales_pd.groupby("customer_id").agg(total_spending=("sales_amount", "sum"))
customer_spending_pd["rank"] = customer_spending_pd["total_spending"].rank(method="dense", ascending=False)
customer_spending_pd: pd.DataFrame = customer_spending_pd.sort_values("rank").reset_index()
print(f"Customer Spending Summary: {len(customer_spending_pd)}")
print(customer_spending_pd.head(5))
print(customer_spending_pd.head(5).to_markdown())
```

<div class="result" markdown>

```txt
Customer Spending Summary: 61
```

```txt
   customer_id  total_spending  rank
0           15         2297.55   1.0
1            4         2237.49   2.0
2           62         2177.35   3.0
3           60         2086.09   4.0
4           21         2016.95   5.0
```

|      | customer_id | total_spending | rank |
| ---: | ----------: | -------------: | ---: |
|    0 |          15 |        2297.55 |    1 |
|    1 |           4 |        2237.49 |    2 |
|    2 |          62 |        2177.35 |    3 |
|    3 |          60 |        2086.09 |    4 |
|    4 |          21 |        2016.95 |    5 |

</div>

### SQL

In SQL, we can use the `DENSE_RANK()` window function to rank values in a query. This function assigns a rank to each row within a partition of a given result set, with no gaps in the ranking values. Note that this function can only be used in congunction with the `OVER` clause, which defines it as a Window function. The `ORDER BY` clause within the `OVER` clause specifies the order in which the rows are ranked.

```python {.sql linenums="1" title="Rank customers by total spending"}
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

### PySpark

In PySpark, we can use the [`F.dense_rank()`][pyspark-dense_rank] function in combination with the [`Window()`][pyspark-window] class to rank values in a DataFrame. The [`Window()`][pyspark-window] class allows us to define a window specification for the calculation, and the [`F.dense_rank()`][pyspark-dense_rank] function calculates the dense rank of each row within that window.

```python {.pyspark linenums="1" title="Rank customers by total spending"}
customer_spending_ps: psDataFrame = (
    df_sales_ps.groupBy("customer_id")
    .agg(F.sum("sales_amount").alias("total_spending"))
    .withColumn("rank", F.dense_rank().over(Window.orderBy(F.desc("total_spending"))))
    .orderBy("rank")
)
print(f"Customer Spending Summary: {customer_spending_ps.count()}")
customer_spending_ps.show(5)
print(customer_spending_ps.limit(5).toPandas().to_markdown())
```

<div class="result" markdown>

```txt
Customer Spending Summary: 61
```

```txt
+-----------+------------------+----+
|customer_id|    total_spending|rank|
+-----------+------------------+----+
|         15|           2297.55|   1|
|          4|           2237.49|   2|
|         62|           2177.35|   3|
|         60|2086.0899999999997|   4|
|         21|           2016.95|   5|
+-----------+------------------+----+
```

|      | customer_id | total_spending | rank |
| ---: | ----------: | -------------: | ---: |
|    0 |          15 |        2297.55 |    1 |
|    1 |           4 |        2237.49 |    2 |
|    2 |          62 |        2177.35 |    3 |
|    3 |          60 |        2086.09 |    4 |
|    4 |          21 |        2016.95 |    5 |

</div>

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

### Pandas

In Pandas it is first necessary to group the sales data by `category` and `product_id`, then aggregate the data for the [`.sum()`][pandas-groupby-sum] of the `quantity` to find the `total_quantity` sold for each product. After that, we can use the [`.rank()`][pandas-rank] method to rank the products within each category based on the total quantity sold.

It is important to note here that we are implementing the [`.rank()`][pandas-rank] method from within an [`.assign()`][pandas-assign] method. This is a common pattern in Pandas to create new columns on a DataFrame based on other columns already existing on the DataFrame, while keeping the DataFrame immutable. Here, we are using the [`.groupby()`][pandas-groupby] method to group the DataFrame by `category`, and then applying the [`.rank()`][pandas-rank] method to the `total_quantity` column within each category. In this way, we are creating a partitioned DataFrame that ranks products by quantity sold within each category.

```python {.pandas linenums="1" title="Rank products by quantity sold, by category"}
product_popularity_pd: pd.DataFrame = (
    df_sales_pd.groupby(["category", "product_id"])
    .agg(
        total_quantity=("quantity", "sum"),
    )
    .reset_index()
    .assign(
        rank=lambda x: x.groupby("category")["total_quantity"].rank(method="dense", ascending=False),
    )
    .sort_values(["rank", "category"])
    .reset_index(drop=True)
)
print(f"Product Popularity Summary: {len(product_popularity_pd)}")
print(product_popularity_pd.head(10))
print(product_popularity_pd.head(10).to_markdown())
```

<div class="result" markdown>

```txt
Product Popularity Summary: 78
```

```txt
      category  product_id  total_quantity  rank
0        Books          11              14   1.0
1     Clothing           7               9   1.0
2  Electronics          37              16   1.0
3         Food          45              34   1.0
4         Home           3              10   1.0
5        Books          28               9   2.0
6     Clothing          35               8   2.0
7  Electronics          35              11   2.0
8         Food           1              16   2.0
9         Home           9               5   2.0
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
|    9 | Home        |          9 |              5 |    2 |

</div>

### SQL

In SQL, we can use the `RANK()` window function to rank products within each category based on the total quantity sold. The `PARTITION BY` clause allows us to partition the data by `category`, and the `ORDER BY` clause specifies the order in which the rows are ranked within each partition.

```python {.sql linenums="1" title="Rank products by quantity sold, by category"}
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

### PySpark

In PySpark, we can use the [`F.dense_rank()`][pyspark-dense_rank] function in combination with the [`Window()`][pyspark-window] class to rank products within each category based on the total quantity sold. We can define the partitioning by using the [`.partitionBy()`][pyspark-window-partitionby] method and parse'ing in the `"category"` column. We can then define the ordering by using the [`.orderBy()`][pyspark-window-orderby] method and parse'ing in the `"total_quantity"` expression to order the products by total quantity sold in descending order with the [`F.desc()`][pyspark-desc] method.

Here, we have also provided an alternative way to define the rank by using the Spark SQL method. The outcome is the same, it's simply written in a SQL-like expression.

```python {.pyspark linenums="1" title="Rank products by quantity sold, by category"}
product_popularity_ps: psDataFrame = (
    df_sales_ps.groupBy("category", "product_id")
    .agg(F.sum("quantity").alias("total_quantity"))
    .withColumns(
        {
            "rank_p": F.dense_rank().over(Window.partitionBy("category").orderBy(F.desc("total_quantity"))),
            "rank_s": F.expr("DENSE_RANK() OVER (PARTITION BY category ORDER BY total_quantity DESC)"),
        }
    )
    .orderBy("rank_p")
)
print(f"Product Popularity Summary: {product_popularity_ps.count()}")
product_popularity_ps.show(10)
print(product_popularity_ps.limit(10).toPandas().to_markdown())
```

<div class="result" markdown>

```txt
Product Popularity Summary: 78
```

```txt
+-----------+----------+--------------+------+------+
|   category|product_id|total_quantity|rank_p|rank_s|
+-----------+----------+--------------+------+------+
|   Clothing|         7|             9|     1|     1|
|      Books|        11|            14|     1|     1|
|Electronics|        37|            16|     1|     1|
|       Food|        45|            34|     1|     1|
|       Home|         3|            10|     1|     1|
|      Books|        28|             9|     2|     2|
|Electronics|        35|            11|     2|     2|
|       Home|        29|             5|     2|     2|
|       Home|        48|             5|     2|     2|
|       Home|         9|             5|     2|     2|
+-----------+----------+--------------+------+------+
only showing top 10 rows
```

|      | category    | product_id | total_quantity | rank_p | rank_s |
| ---: | :---------- | ---------: | -------------: | -----: | -----: |
|    0 | Clothing    |          7 |              9 |      1 |      1 |
|    1 | Books       |         11 |             14 |      1 |      1 |
|    2 | Electronics |         37 |             16 |      1 |      1 |
|    3 | Food        |         45 |             34 |      1 |      1 |
|    4 | Home        |          3 |             10 |      1 |      1 |
|    5 | Books       |         28 |              9 |      2 |      2 |
|    6 | Clothing    |         35 |              8 |      2 |      2 |
|    7 | Electronics |         35 |             11 |      2 |      2 |
|    8 | Food        |          1 |             16 |      2 |      2 |
|    9 | Home        |         29 |              5 |      2 |      2 |

</div>

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
