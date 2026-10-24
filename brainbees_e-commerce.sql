CREATE DATABASE Brainbees_Ecommerce;
USE Brainbees_Ecommerce;

-- First verify stg_tables
SELECT COUNT(*) AS row_count FROM stg_Customers;

SELECT COUNT(*) AS row_count FROM stg_Categories;

SELECT COUNT(*) AS row_count FROM stg_Products;

SELECT COUNT(*) AS row_count FROM stg_Orders;

SELECT COUNT(*) AS row_count FROM stg_Order_Items;

SELECT COUNT(*) AS row_count FROM stg_Payments;

SELECT COUNT(*) AS row_count FROM stg_Shipping;

SELECT COUNT(*) AS row_count FROM stg_Reviews;

SELECT * FROM stg_Customers LIMIT 10;
SELECT * FROM stg_Categories LIMIT 10;
SELECT * FROM stg_Products LIMIT 10;
SELECT * FROM stg_Orders LIMIT 10;
SELECT * FROM stg_Order_Items LIMIT 10;
SELECT * FROM stg_Payments LIMIT 10;
SELECT * FROM stg_Shipping LIMIT 10;
SELECT * FROM stg_Reviews LIMIT 10;

-- Data Quality Checks 
-- Customer Validation
SELECT
    ï»¿Customer_ID,
    COUNT(*) AS duplicate_count
FROM stg_Customers
GROUP BY ï»¿Customer_ID
HAVING COUNT(*) > 1;

SELECT *
FROM stg_Customers
WHERE ï»¿Customer_ID IS NULL
   OR TRIM(ï»¿Customer_ID) = '';

SELECT *
FROM stg_Customers
WHERE First_Name AND Last_Name IS NULL
   OR TRIM(First_Name) = '' AND TRIM(Last_Name) = '' ;

SELECT *
FROM stg_Customers
WHERE Signup_Date > CURRENT_DATE();

SELECT *
FROM stg_Customers
WHERE Date_of_Birth > CURRENT_DATE();


SELECT
    COUNT(*) AS total_customers,
    COUNT(DISTINCT Customer_ID) AS unique_customers,
    COUNT(DISTINCT City) AS cities,
    COUNT(DISTINCT State) AS states
FROM stg_Customers;

-- Categories Validation
SELECT
    ï»¿Category_ID,
    COUNT(*) AS duplicate_count
FROM stg_Categories
GROUP BY ï»¿Category_ID
HAVING COUNT(*) > 1;

SELECT *
FROM stg_Categories
WHERE ï»¿Category_ID IS NULL
   OR TRIM(ï»¿Category_ID) = '';

SELECT
    Department,
    COUNT(*) AS category_count
FROM stg_Categories
GROUP BY Department
ORDER BY category_count DESC;

-- Products Validation
SELECT
    ï»¿Product_ID,
    COUNT(*) AS duplicate_count
FROM stg_Products
GROUP BY ï»¿Product_ID
HAVING COUNT(*) > 1;

SELECT *
FROM stg_Products
WHERE Selling_Price > MRP;

SELECT *
FROM stg_Products
WHERE Cost_Price >= Selling_Price;

SELECT *
FROM stg_Products
WHERE Cost_Price < 0
   OR Selling_Price < 0
   OR MRP < 0;

SELECT *
FROM stg_Products
WHERE Product_Launch_Date > CURRENT_DATE();

ALTER TABLE stg_products
CHANGE COLUMN `ï»¿Product_ID` Product_ID TEXT;

ALTER TABLE stg_categories
CHANGE COLUMN `ï»¿Category_ID` Category_ID TEXT;

SELECT
    p.Product_ID,
    p.Category_ID
FROM stg_products p
LEFT JOIN stg_categories c
    ON p.Category_ID = c.Category_ID
WHERE c.Category_ID IS NULL;

-- Orders validation
SELECT
    ï»¿Order_ID,
    COUNT(*) AS duplicate_count
FROM stg_Orders
GROUP BY ï»¿Order_ID
HAVING COUNT(*) > 1;

ALTER TABLE stg_Orders
CHANGE COLUMN `ï»¿Order_ID` Order_ID TEXT;

ALTER TABLE stg_Customers
CHANGE COLUMN `ï»¿Customer_ID` Customer_ID TEXT;

SELECT
    o.Order_ID,
    o.Customer_ID
FROM stg_Orders o
LEFT JOIN stg_Customers c
    ON o.Customer_ID = c.Customer_ID
WHERE c.Customer_ID IS NULL;

SELECT *
FROM stg_Orders
WHERE Order_Date > CURRENT_DATE();

SELECT *
FROM stg_Orders
WHERE Order_Total < 0;

SELECT
    Order_Status,
    COUNT(*) AS order_count
FROM stg_Orders
GROUP BY Order_Status
ORDER BY order_count DESC;

SELECT
    Order_Channel,
    COUNT(*) AS total_orders,
    SUM(Order_Total) AS total_revenue,
    AVG(Order_Total) AS average_order_value
FROM stg_Orders
GROUP BY Order_Channel
ORDER BY total_revenue DESC;

-- Order Items validation
SELECT
    ï»¿Order_Item_ID,
    COUNT(*) AS duplicate_count
FROM stg_Order_Items
GROUP BY ï»¿Order_Item_ID
HAVING COUNT(*) > 1;

SELECT *
FROM stg_Order_Items
WHERE Quantity <= 0;

SELECT *
FROM stg_Order_Items
WHERE Unit_Price < 0
   OR Discount_Amount < 0
   OR Line_Total < 0
   OR Profit_Amount < 0;

ALTER TABLE stg_Order_Items
CHANGE COLUMN `ï»¿Order_Item_ID` Order_Item_ID TEXT;

SELECT oi.*
FROM stg_Order_Items oi
LEFT JOIN stg_Orders o
    ON oi.Order_Item_ID = o.Order_ID
WHERE o.Order_ID IS NULL;

SELECT oi.*
FROM stg_Order_Items oi
LEFT JOIN stg_Products p
    ON oi.Product_ID = p.Product_ID
WHERE p.Product_ID IS NULL;

-- Payments validation

SELECT
    ï»¿Payment_ID,
    COUNT(*) AS duplicate_count
FROM stg_Payments
GROUP BY ï»¿Payment_ID
HAVING COUNT(*) > 1;


SELECT p.*
FROM stg_Payments p
LEFT JOIN stg_Orders o
    ON p.Order_ID = o.Order_ID
WHERE o.Order_ID IS NULL;


SELECT *
FROM stg_Payments
WHERE Amount < 0;


SELECT *
FROM stg_Payments
WHERE Payment_Date > CURRENT_DATE();


SELECT
    Payment_Method,
    COUNT(*) AS transaction_count,
    SUM(Amount) AS total_amount
FROM stg_Payments
GROUP BY Payment_Method
ORDER BY total_amount DESC;

-- Shipping Validation

SELECT
    ï»¿Shipping_ID,
    COUNT(*) AS duplicate_count
FROM stg_Shipping
GROUP BY ï»¿Shipping_ID
HAVING COUNT(*) > 1;


SELECT s.*
FROM stg_Shipping s
LEFT JOIN stg_Orders o
    ON s.Order_ID = o.Order_ID
WHERE o.Order_ID IS NULL;


SELECT
    s.*
FROM stg_Shipping s
JOIN stg_Orders o
    ON s.Order_ID = o.Order_ID
WHERE s.Shipping_Date < o.Order_Date;


SELECT *
FROM stg_Shipping
WHERE Expected_Delivery_Date < Shipping_Date;


SELECT *
FROM stg_Shipping
WHERE Actual_Delivery_Date IS NOT NULL
  AND Actual_Delivery_Date < Shipping_Date;

SELECT *
FROM stg_Shipping
WHERE Return_Pickup_Date IS NOT NULL
  AND Actual_Delivery_Date IS NOT NULL
  AND Return_Pickup_Date < Actual_Delivery_Date;

-- Reviews Validation

SELECT
    ï»¿Review_ID,
    COUNT(*) AS duplicate_count
FROM stg_Reviews
GROUP BY ï»¿Review_ID
HAVING COUNT(*) > 1;

SELECT *
FROM stg_Reviews
WHERE Rating NOT BETWEEN 1 AND 5;

SELECT r.*
FROM stg_Reviews r
LEFT JOIN stg_Customers c
    ON r.Customer_ID = c.Customer_ID
WHERE c.Customer_ID IS NULL;


SELECT r.*
FROM stg_Reviews r
LEFT JOIN stg_Products p
    ON r.Product_ID = p.Product_ID
WHERE p.Product_ID IS NULL;


SELECT r.*
FROM stg_Reviews r
LEFT JOIN stg_Orders o
    ON r.Order_ID = o.Order_ID
WHERE o.Order_ID IS NULL;

-- Create the production tables
-- Dimension: Customers

SELECT Customer_ID,
       CONCAT(First_Name, ' ', Last_Name) AS Customer_Names
FROM stg_Customers;

CREATE TABLE dim_Customers AS
SELECT DISTINCT
    Customer_ID,
    First_Name,
    Last_Name,
    Gender,
    Date_of_Birth,
    City,
    State,
    Customer_Segment,
    Loyalty_Tier,
    Signup_Date,
    Email,
    Phone
FROM stg_Customers;

SELECT *
FROM dim_Customers;

-- Dimension: Categories
CREATE TABLE dim_Categories AS
SELECT DISTINCT
    Category_ID,
    Category_Name,
    Department
FROM stg_Categories;

SELECT *
FROM dim_Categories ;

-- Dimension: Products
CREATE TABLE dim_Products AS
SELECT DISTINCT
    Product_ID,
    Product_Name,
    Category_ID,
    Brand,
    Cost_Price,
    Selling_Price,
    MRP,
    Product_Launch_Date
FROM stg_Products;

SELECT * FROM dim_Products;
-- Fact: Orders
CREATE TABLE fact_Orders AS
SELECT
    Order_ID,
    Customer_ID,
    Order_Date,
    Order_Status,
    Order_Channel,
    Order_Total,
    Payment_Status
FROM stg_Orders;

SELECT * FROM fact_Orders;
-- Fact: Order Items
CREATE TABLE fact_Order_Items AS
SELECT
    Order_Item_ID,
    Order_ID,
    Product_ID,
    Quantity,
    Unit_Price,
    Discount_Amount,
    Discount_Percentage,
    Line_Total,
    Profit_Amount
FROM stg_Order_Items;

SELECT * FROM fact_Order_Items;

-- Fact: Payments
CREATE TABLE fact_Payments AS
SELECT
    ï»¿Payment_ID,
    Order_ID,
    Payment_Date,
    Payment_Method,
    Payment_Status,
    Amount,
    Transaction_ID
FROM stg_Payments;

SELECT * FROM fact_Payments;

-- Fact: Shipping
CREATE TABLE fact_Shipping AS
SELECT
    ï»¿Shipping_ID,
    Order_ID,
    Shipping_Date,
    Expected_Delivery_Date,
    Actual_Delivery_Date,
    Return_Pickup_Date,
    Shipping_Status,
    Courier_Partner
FROM stg_Shipping;

SELECT * FROM fact_Shipping;

-- Fact: Reviews
CREATE TABLE fact_Reviews AS
SELECT
    ï»¿Review_ID,
    Customer_ID,
    Product_ID,
    Order_ID,
    Rating,
    Review_Text,
    Review_Date
FROM stg_Reviews;

SELECT * FROM fact_Reviews;

-- ADD INDEXES
ALTER TABLE fact_Orders MODIFY Order_ID VARCHAR(10) NOT NULL;
CREATE INDEX idx_orders_customer
ON fact_Orders(Order_ID); 

SELECT * FROM fact_Orders;

ALTER TABLE fact_Order_Items MODIFY Order_ID VARCHAR(10) NOT NULL;
CREATE INDEX idx_order_items_order
ON fact_Order_Items(Order_ID);

SELECT * FROM fact_Order_Items;

ALTER TABLE fact_Order_Items MODIFY Product_ID VARCHAR(10) NOT NULL;
CREATE INDEX idx_order_items_product
ON fact_Order_Items(Product_ID);

SELECT * FROM fact_Order_Items;

ALTER TABLE fact_Payments MODIFY Order_ID VARCHAR(10) NOT NULL;
CREATE INDEX idx_payments_order
ON fact_Payments(Order_ID);

SELECT * FROM fact_Payments;

ALTER TABLE fact_Shipping MODIFY Order_ID VARCHAR(10) NOT NULL;
CREATE INDEX idx_shipping_order
ON fact_Shipping(Order_ID);

SELECT * FROM fact_Shipping;

ALTER TABLE fact_Reviews MODIFY Customer_ID VARCHAR(10) NOT NULL;
CREATE INDEX idx_reviews_customer
ON fact_Reviews(Customer_ID);

SELECT * FROM fact_Reviews;

ALTER TABLE fact_reviews MODIFY Product_ID VARCHAR(10) NOT NULL;
CREATE INDEX idx_reviews_product
ON fact_Reviews(Product_ID);

SELECT * FROM fact_Reviews;

-- Create your analytical SQL views

-- View 1 — Sales overview
CREATE OR REPLACE VIEW vw_Sales_Overview AS
SELECT
    o.Order_ID,
    o.Order_Date,
    o.Customer_ID,
    oi.Product_ID,
    oi.Quantity,
    oi.Unit_Price,
    oi.Discount_Amount,
    oi.Line_Total,
    oi.Profit_Amount,
    o.Order_Status,
    o.Order_Channel,
    o.Payment_Status
FROM fact_Orders o
JOIN fact_Order_Items oi
    ON o.Order_ID = oi.Order_ID;

SELECT * FROM vw_Sales_Overview;

-- View 2 — Product performance
CREATE OR REPLACE VIEW vw_Product_Performance AS
SELECT
    p.Product_ID,
    p.Product_Name,
    p.Brand,
    c.Category_Name,
    c.Department,
    COUNT(DISTINCT oi.Order_ID) AS Total_Orders,
    SUM(oi.Quantity) AS Units_Sold,
    SUM(oi.Line_Total) AS Revenue,
    SUM(oi.Profit_Amount) AS Profit
FROM dim_Products p
LEFT JOIN dim_Categories c
    ON p.Category_ID = c.Category_ID
LEFT JOIN fact_Order_Items oi
    ON p.Product_ID = oi.Product_ID
GROUP BY
    p.Product_ID,
    p.Product_Name,
    p.Brand,
    c.Category_Name,
    c.Department;

SELECT * FROM vw_Product_Performance;

-- Customer Performance
CREATE OR REPLACE VIEW vw_Customer_Performance AS
SELECT
    c.Customer_ID,
    c.First_Name,
    c.Last_Name,
    c.Gender,
    c.City,
    c.State,
    c.Customer_Segment,
    c.Loyalty_Tier,
    COUNT(DISTINCT o.Order_ID) AS Total_Orders,
    COALESCE(SUM(o.Order_Total), 0) AS Total_Spending,
    COALESCE(AVG(o.Order_Total), 0) AS Average_Order_Value
FROM dim_Customers c
LEFT JOIN fact_Orders o
    ON c.Customer_ID = o.Customer_ID
GROUP BY
    c.Customer_ID,
    c.First_Name,
    c.Last_Name,
    c.Gender,
    c.City,
    c.State,
    c.Customer_Segment,
    c.Loyalty_Tier;

SELECT * FROM vw_Customer_Performance;

-- View 4 — Category performance
CREATE OR REPLACE VIEW vw_Category_Performance AS
SELECT
    c.Category_ID,
    c.Category_Name,
    c.Department,
    COUNT(DISTINCT p.Product_ID) AS Product_Count,
    COALESCE(SUM(oi.Quantity), 0) AS Units_Sold,
    COALESCE(SUM(oi.Line_Total), 0) AS Revenue,
    COALESCE(SUM(oi.Profit_Amount), 0) AS Profit
FROM dim_Categories c
LEFT JOIN dim_Products p
    ON c.Category_ID = p.Category_ID
LEFT JOIN fact_Order_Items oi
    ON p.Product_ID = oi.Product_ID
GROUP BY
    c.Category_ID,
    c.Category_Name,
    c.Department;

SELECT * FROM vw_Category_Performance;

-- View 5 — Monthly sales
SELECT
    COUNT(*) AS Total_Rows,
    COUNT(Order_ID) AS Order_IDs,
    COUNT(Order_Date) AS Valid_Order_Dates
FROM fact_Orders;

SELECT
    Order_ID,
    Order_Date
FROM fact_Orders;

SELECT
    COUNT(*) AS Total_Rows,
    COUNT(Order_ID) AS Order_IDs,
    COUNT(Quantity) AS Quantities,
    COUNT(Line_Total) AS Line_Totals,
    COUNT(Profit_Amount) AS Profits
FROM fact_Order_Items;

SELECT
    COUNT(*) AS Matching_Rows
FROM fact_Orders o
JOIN fact_Order_Items oi
    ON o.Order_ID = oi.Order_ID;
    
SELECT
    Order_Item_ID,
    Order_ID,
    Product_ID,
    Quantity,
    Line_Total,
    Profit_Amount
FROM fact_Order_Items;

SHOW COLUMNS FROM fact_orders;

SELECT
    Order_Date
FROM fact_orders
WHERE Order_Date IS NOT NULL
  AND STR_TO_DATE(Order_Date, '%Y-%m-%d') IS NULL
LIMIT 20;

SELECT
    Order_ID,
    Order_Date
FROM fact_orders
LIMIT 10;

ALTER TABLE fact_orders
ADD COLUMN Order_Date_New DATE;

UPDATE fact_orders
SET Order_Date_New = STR_TO_DATE(Order_Date, '%d-%m-%Y')
WHERE Order_ID IS NOT NULL;

SET SQL_SAFE_UPDATES = 0;

UPDATE fact_orders
SET Order_Date_New = STR_TO_DATE(Order_Date, '%d-%m-%Y');

SELECT
    Order_ID,
    Order_Date,
    Order_Date_New
FROM fact_orders
LIMIT 10;

SELECT
    COUNT(*) AS Failed_Conversions
FROM fact_orders
WHERE Order_Date IS NOT NULL
  AND Order_Date_New IS NULL;

ALTER TABLE fact_orders
DROP COLUMN Order_Date;

ALTER TABLE fact_orders
RENAME COLUMN Order_Date_New TO Order_Date;

SET SQL_SAFE_UPDATES = 1;

SHOW COLUMNS FROM fact_orders;

SELECT
    YEAR(o.Order_Date) AS Order_Year,
    MONTH(o.Order_Date) AS Order_Month,
    DATE_FORMAT(o.Order_Date, '%Y-%m') AS Year_Month_Key,
    COUNT(DISTINCT o.Order_ID) AS Total_Orders,
    SUM(oi.Quantity) AS Units_Sold,
    SUM(oi.Line_Total) AS Revenue,
    SUM(oi.Profit_Amount) AS Profit
FROM fact_orders o
JOIN fact_order_items oi
    ON o.Order_ID = oi.Order_ID
GROUP BY
    YEAR(o.Order_Date),
    MONTH(o.Order_Date),
    DATE_FORMAT(o.Order_Date, '%Y-%m')
ORDER BY
    Order_Year,
    Order_Month;

CREATE OR REPLACE VIEW vw_Monthly_Sales AS
SELECT
    YEAR(o.Order_Date) AS Order_Year,
    MONTH(o.Order_Date) AS Order_Month,
    DATE_FORMAT(o.Order_Date, '%Y-%m') AS Year_Month_Key,
    
    COUNT(DISTINCT o.Order_ID) AS Total_Orders,
    SUM(oi.Quantity) AS Units_Sold,
    SUM(oi.Line_Total) AS Revenue,
    SUM(oi.Profit_Amount) AS Profit

FROM fact_orders o

INNER JOIN fact_order_items oi
    ON o.Order_ID = oi.Order_ID

GROUP BY
    YEAR(o.Order_Date),
    MONTH(o.Order_Date),
    DATE_FORMAT(o.Order_Date, '%Y-%m');

SELECT * FROM vw_monthly_sales;

-- View 6 — Customer + order analysis
CREATE OR REPLACE VIEW vw_Customer_Order_Analysis AS
SELECT
    c.Customer_ID,
    c.First_Name,
    c.Last_Name,
    c.Customer_Segment,
    c.Loyalty_Tier,
    c.City,
    c.State,
    o.Order_ID,
    o.Order_Date,
    o.Order_Status,
    o.Order_Channel,
    o.Order_Total
FROM dim_Customers c
JOIN fact_Orders o
    ON c.Customer_ID = o.Customer_ID;
    
SELECT * FROM vw_Customer_Order_Analysis;

-- View 7 — Reviews analysis
CREATE OR REPLACE VIEW vw_Product_Reviews AS
SELECT
    p.Product_ID,
    p.Product_Name,
    c.Category_Name,
    AVG(r.Rating) AS Average_Rating,
    COUNT(r.ï»¿Review_ID) AS Review_Count
FROM dim_Products p
LEFT JOIN dim_Categories c
    ON p.Category_ID = c.Category_ID
LEFT JOIN fact_Reviews r
    ON p.Product_ID = r.Product_ID
GROUP BY
    p.Product_ID,
    p.Product_Name,
    c.Category_Name;
    
SELECT * FROM vw_Product_Reviews;

-- Executive KPI query
SELECT
    COUNT(DISTINCT o.Order_ID) AS Total_Orders,
    COUNT(DISTINCT o.Customer_ID) AS Active_Customers,
    SUM(oi.Quantity) AS Units_Sold,
    SUM(oi.Line_Total) AS Total_Revenue,
    SUM(oi.Profit_Amount) AS Total_Profit,
    AVG(o.Order_Total) AS Average_Order_Value
FROM fact_Orders o
JOIN fact_Order_Items oi
    ON o.Order_ID = oi.Order_ID;

-- Profit margin
SELECT
    SUM(oi.Line_Total) AS Revenue,
    SUM(oi.Profit_Amount) AS Profit,
    ROUND(
        SUM(oi.Profit_Amount) /
        NULLIF(SUM(oi.Line_Total), 0) * 100,
        2
    ) AS Profit_Margin_Percent
FROM fact_Order_Items oi;

-- Top 10 products
SELECT
    p.Product_ID,
    p.Product_Name,
    SUM(oi.Quantity) AS Units_Sold,
    SUM(oi.Line_Total) AS Revenue,
    SUM(oi.Profit_Amount) AS Profit
FROM fact_Order_Items oi
JOIN dim_Products p
    ON oi.Product_ID = p.Product_ID
GROUP BY
    p.Product_ID,
    p.Product_Name
ORDER BY Revenue DESC
LIMIT 10;

-- Top customers
SELECT
    c.Customer_ID,
    c.First_Name,
    c.Last_Name,
    COUNT(DISTINCT o.Order_ID) AS Total_Orders,
    SUM(o.Order_Total) AS Total_Spending
FROM dim_Customers c
JOIN fact_Orders o
    ON c.Customer_ID = o.Customer_ID
GROUP BY
    c.Customer_ID,
    c.First_Name,
    c.Last_Name
ORDER BY Total_Spending DESC
LIMIT 10;

-- Monthly growth
WITH monthly AS
(
    SELECT
        DATE_FORMAT(o.Order_Date, '%Y-%m') AS Year_Month_Key,
        SUM(oi.Line_Total) AS Revenue
    FROM fact_Orders o
    JOIN fact_Order_Items oi
        ON o.Order_ID = oi.Order_ID
    GROUP BY DATE_FORMAT(o.Order_Date, '%Y-%m')
)

SELECT
    Year_Month_Key,
    Revenue,
    LAG(Revenue) OVER (
        ORDER BY Year_Month_Key
    ) AS Previous_Month_Revenue,

    ROUND(
        (
            Revenue -
            LAG(Revenue) OVER (
                ORDER BY Year_Month_Key
            )
        )
        /
        NULLIF(
            LAG(Revenue) OVER (
                ORDER BY Year_Month_Key
            ),
            0
        ) * 100,
        2
    ) AS MoM_Growth_Percent

FROM monthly
ORDER BY Year_Month_Key;

-- Ranking Products
WITH product_sales AS
(
    SELECT
        p.Product_ID,
        p.Product_Name,
        SUM(oi.Line_Total) AS Revenue
    FROM dim_Products p
    JOIN fact_Order_Items oi
        ON p.Product_ID = oi.Product_ID
    GROUP BY
        p.Product_ID,
        p.Product_Name
)

SELECT
    Product_ID,
    Product_Name,
    Revenue,
    DENSE_RANK() OVER (
        ORDER BY Revenue DESC
    ) AS Revenue_Rank
FROM product_sales;




