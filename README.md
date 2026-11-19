# 🛒 Brainbees E-Commerce Business Intelligence & Machine Learning Platform

An end-to-end E-Commerce Analytics project that transforms raw business data into meaningful insights using **Python, SQL, MySQL, Power BI, and Machine Learning**.

The project focuses on analysing e-commerce business operations, including customers, products, orders, payments, shipping and reviews. It combines data preparation, database management, business intelligence dashboards and predictive analytics to support data-driven decision-making.

---

## 📌 Project Overview

E-commerce businesses generate large amounts of data across different business operations. Analysing this information separately makes it difficult to understand overall business performance.

The objective of this project is to build an integrated analytics solution that:

- Organises e-commerce data into structured database tables.
- Cleans and validates raw datasets.
- Stores and manages data using MySQL.
- Uses SQL queries and analytical views to extract business insights.
- Develops interactive Power BI dashboards.
- Explores machine learning for sales forecasting.
- Exposes forecasting functionality through a FastAPI application.
- Uses Docker and Render for application deployment.

---

## 🎯 Project Objectives

1. Build a structured e-commerce database.
2. Perform data cleaning and quality validation.
3. Establish relationships between business tables.
4. Analyse sales, revenue and customer behaviour.
5. Develop interactive business intelligence dashboards.
6. Explore predictive analytics using machine learning.
7. Make forecasting functionality accessible through an API.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Excel | Initial dataset preparation |
| Power Query | Data cleaning and transformation |
| Python | Data processing and automation |
| Pandas | Data manipulation and validation |
| MySQL | Relational database management |
| SQL | Data extraction, joins and analysis |
| Power BI | Interactive dashboards and reporting |
| DAX | Business metrics and calculations |
| Machine Learning | Predictive analytics and forecasting |
| FastAPI | Model/API integration |
| Docker | Application containerisation |
| Render | API deployment |
| n8n | Workflow automation exploration |
| Git & GitHub | Version control and project management |

---

## 🗄️ Database Structure

The project uses eight core business tables.

| Table Name | Description |
|---|---|
| Customers | Customer information |
| Categories | Product category details |
| Products | Product information |
| Orders | Order transaction details |
| Order_Items | Individual products within orders |
| Payments | Payment transaction information |
| Shipping | Shipping and delivery information |
| Reviews | Customer product reviews |

### Database Relationships

The database connects related business entities using primary keys and foreign keys.

Main relationships include:

- Customers → Orders
- Orders → Order_Items
- Order_Items → Products
- Categories → Products
- Orders → Payments
- Orders → Shipping
- Reviews → Customers / Products

**Data modelling note:** Orders and Order_Items represent transactional data at different levels of detail. Customers, Products and Categories provide descriptive information. Payments, Shipping and Reviews are modelled according to their individual data grain.

---

## 🔄 Project Workflow

The project follows an end-to-end data analytics workflow.

```text
             RAW E-COMMERCE DATA
                      |
                      v
                EXCEL DATASETS
                      |
                      v
           POWER QUERY / PYTHON
                      |
                      v
             DATA CLEANING
                      |
                      v
          DATA QUALITY VALIDATION
                      |
                      v
                CSV FILES
                      |
                      v
            MYSQL STAGING TABLES
                      |
                      v
             MYSQL MAIN TABLES
                      |
                      v
                SQL ANALYSIS
                      |
             +--------+--------+
             |                 |
             v                 v
       POWER BI          MACHINE LEARNING
       DASHBOARDS          FORECASTING
             |                 |
             v                 v
       BUSINESS INSIGHTS   FASTAPI
                               |
                               v
                             DOCKER
                               |
                               v
                             RENDER
                               |
                               v
                      API INTEGRATION
```

---

## 🧹 Data Cleaning & Validation

Data preparation is an important stage of the project.

The datasets are inspected and prepared using Python, Pandas and Power Query.

Key activities include:

- Checking missing and NULL values.
- Identifying duplicate records.
- Standardising column names and text values.
- Handling inconsistent date formats.
- Checking appropriate data types.
- Validating primary and foreign key relationships.
- Identifying invalid references and orphan records.
- Checking business calculations.
- Comparing source and transformed data.

These checks help improve data reliability before performing analysis.

---

## 🗃️ SQL & Database Analysis

MySQL is used to store structured e-commerce data and perform analytical queries.

SQL operations include:

- SELECT and WHERE statements.
- INNER JOIN and other table joins.
- GROUP BY and ORDER BY.
- Aggregate functions such as SUM(), COUNT() and AVG().
- Data validation queries.
- Monthly sales analysis.
- Revenue calculations.
- Analytical SQL views.

Example analytical view:

```sql
CREATE VIEW vw_Monthly_Sales AS
SELECT
    YEAR(Order_Date) AS Sales_Year,
    MONTH(Order_Date) AS Sales_Month,
    COUNT(DISTINCT Order_ID) AS Total_Orders
FROM Orders
GROUP BY
    YEAR(Order_Date),
    MONTH(Order_Date);
```

*This is an illustrative example of monthly order analysis. Adjust the query to match the actual column names and sales calculation used in the database.*

---

## 📊 Power BI Dashboard

Power BI is used as the main business intelligence and visualisation layer.

The dashboards are designed to help understand different aspects of e-commerce performance.

### Key Analysis Areas

- Overall sales performance.
- Revenue trends.
- Monthly sales analysis.
- Product performance.
- Category-wise analysis.
- Customer behaviour.
- Payment analysis.
- Shipping performance.
- Review analysis.

### Important Business KPIs

| KPI | Business Purpose |
|---|---|
| Total Revenue | Measures overall sales value |
| Total Orders | Measures order volume |
| Total Units Sold | Measures product quantity sold |
| Average Order Value | Measures average value per order |
| Total Customers | Measures customer base |
| Profit | Measures business earnings, where profit data is available |
| Profit Margin | Measures profitability, where profit data is available |

DAX measures are used to calculate business metrics and support interactive reporting.

---

## 🤖 Machine Learning & Predictive Analytics

The project extends beyond descriptive analytics by exploring machine learning for predictive business analysis.

The forecasting component is designed to help understand possible future sales trends.

Main activities include:

- Preparing data for forecasting.
- Analysing historical sales patterns.
- Building a machine learning forecasting workflow.
- Integrating forecasting functionality with FastAPI.

The model's actual performance should be evaluated using suitable forecasting metrics before making business predictions.

---

## 🚀 API Development & Deployment

The machine learning forecasting functionality is made accessible through a FastAPI application.

### Deployment Technologies

**FastAPI:** Provides API endpoints for accessing forecasting functionality.

**Docker:** Packages the application and its dependencies into a container.

**Render:** Hosts the deployed API application.

**n8n:** Explored for connecting the API with automated workflows.

### API Endpoint

```text
/all-forecasts
```

The endpoint provides access to the forecasting functionality implemented in the application.

Docker image reference:

```text
anuragmondal698/brainbees-ml-api:v1
```

---

## 📈 Business Value

This project demonstrates how e-commerce data can be used to support business decisions.

Potential business applications include:

- Monitoring sales and revenue performance.
- Identifying high-performing products and categories.
- Understanding customer purchasing behaviour.
- Monitoring payment and shipping information.
- Analysing customer feedback.
- Supporting sales forecasting and planning.
- Improving the availability of business information through dashboards and APIs.

---

## 📂 Repository

GitHub Repository:

[Brainbees E-Commerce BI & ML](https://github.com/anuragmondal698/Brainbees-Ecommerce-BI-ML)

The repository contains the project resources and implementation files. Refer to the repository folders for the available datasets, scripts, SQL files, dashboards and application code.

---

## 👨‍💻 Skills Demonstrated

- Data Analytics
- Data Cleaning and Data Transformation
- Python and Pandas
- SQL and MySQL
- Relational Database Management
- Data Modelling
- Data Quality Validation
- Power BI
- DAX
- Business Intelligence
- Machine Learning
- FastAPI
- Docker
- API Integration
- Workflow Automation
- Git and GitHub

---

## 🔮 Future Enhancements

- Improve forecasting performance through model evaluation and tuning.
- Add more advanced customer segmentation.
- Explore customer churn prediction.
- Develop automated data-refresh workflows.
- Expand dashboard reporting and business KPIs.
- Improve API monitoring and error handling.

---

## 👤 Author

**Anurag Mondal**

MBA – Business Intelligence & Data Analytics  
B.Tech – Electrical Engineering

**Areas of Interest:** Data Analytics | Business Intelligence | SQL | Python | Power BI | Machine Learning

GitHub: [anuragmondal698](https://github.com/anuragmondal698)

---

⭐ If you find this project interesting, feel free to explore the repository.
