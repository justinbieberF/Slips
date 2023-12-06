slip 1// Write a Java Program to implement I/O Decorator for converting uppercase letters to
lower case letters.

Main.java


import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class Main {
    public static void main(String[] args) {
        if (args.length != 1) {
            System.out.println("Usage: java Main <filename>");
            System.exit(1);
        }

        String filename = args[0];

        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String lowercaseLine = line.toLowerCase();
                System.out.println(lowercaseLine);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}









// Write a Python program to prepare Scatter Plot for Iris Dataset.

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create a scatter plot for Sepal Length vs Sepal Width
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0, 0], X[y == 0, 1], label='Setosa', c='b', marker='o')
plt.scatter(X[y == 1, 0], X[y == 1, 1], label='Versicolor', c='g', marker='s')
plt.scatter(X[y == 2, 0], X[y == 2, 1], label='Virginica', c='r', marker='^')

# Set labels and title
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Iris Dataset - Sepal Length vs Sepal Width')

# Add legend
plt.legend(loc='best')

# Show the plot
plt.show()









// Create an HTML form that contain the Student Registration details and write a JavaScript to validate Student first and last name as it should not contain other than alphabets and age should be between 18 to 50.
1st program-
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Details</title>
    <style>
        .error {
            color: red;
        }
    </style>
</head>
<body>

<h2>Employee Details Form</h2>

<form id="employeeForm" onsubmit="return validateForm()">
    <label for="dob">Date of Birth:</label>
    <input type="date" id="dob" name="dob" required>
    <span id="dobError" class="error"></span><br>

    <label for="joiningDate">Joining Date:</label>
    <input type="date" id="joiningDate" name="joiningDate" required>
    <span id="joiningDateError" class="error"></span><br>

    <label for="salary">Salary:</label>
    <input type="number" id="salary" name="salary" required>
    <span id="salaryError" class="error"></span><br>

    <input type="submit" value="Submit">
</form>

<script>
    function validateForm() {
        // Get form inputs
        var dob = new Date(document.getElementById('dob').value);
        var joiningDate = new Date(document.getElementById('joiningDate').value);
        var salary = parseFloat(document.getElementById('salary').value);

        // Validate Date of Birth
        if (isNaN(dob.getTime())) {
            document.getElementById('dobError').innerHTML = 'Invalid Date of Birth';
            return false;
        } else {
            document.getElementById('dobError').innerHTML = '';
        }

        // Validate Joining Date
        if (isNaN(joiningDate.getTime())) {
            document.getElementById('joiningDateError').innerHTML = 'Invalid Joining Date';
            return false;
        } else {
            document.getElementById('joiningDateError').innerHTML = '';
        }

        // Validate Salary (positive number)
        if (isNaN(salary) || salary <= 0) {
            document.getElementById('salaryError').innerHTML = 'Salary should be a positive number';
            return false;
        } else {
            document.getElementById('salaryError').innerHTML = '';
        }

        // If all validations pass, the form is valid
        return true;
    }
</script>

</body>
</html>


slip 2// Write a Java Program to implement Singleton pattern for multithreading.


public class Question2 {
public static void main(String ar[]) {
Test1 t = new Test1();
Test1 t2 = new Test1();
Test1 t3 = new Test1();
Thread tt = new Thread(t);
Thread tt2 = new Thread(t2);
Thread tt3 = new Thread(t3);
Thread tt4 = new Thread(t);
Thread tt5 = new Thread(t);
tt.start();
tt2.start();
tt3.start();
tt4.start();
tt5.start();
}
}
final class Test1 implements Runnable {
@Override
public void run() {
for (int i = 0; i < 5; i++) {
System.out.println(Thread.currentThread().getName() + " : " +
Single.getInstance().hashCode());
}
}
}
class Single {
private final static Single sing = new Single();
private Single() {
}
public static Single getInstance() {
return sing;
}
}






// Write a python program to find all null values in a given dataset and remove them 

import pandas as pd

# Sample dataset
data = {
    'Name': ['Alice', 'Bob', 'Charlie', None, 'Eve'],
    'Age': [25, 30, None, 28, 22],
    'City': [None, 'New York', 'Los Angeles', 'Chicago', 'San Francisco']
}

# Create a DataFrame from the sample data
df = pd.DataFrame(data)

# Find and display the rows with null values
rows_with_null = df[df.isnull().any(axis=1)]
print("Rows with null values:")
print(rows_with_null)

# Remove rows with null values
df_cleaned = df.dropna()

# Display the cleaned DataFrame
print("\nDataFrame after removing null values:")
print(df_cleaned)






slip2// Create an HTML form that contain the Employee Registration details and write 
a JavaScript to validate DOB, Joining Date, and Salary. 

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Details</title>
    <style>
        .error {
            color: red;
        }
    </style>
</head>
<body>

<h2>Employee Details Form</h2>

<form id="employeeForm" onsubmit="return validateForm()">
    <label for="dob">Date of Birth:</label>
    <input type="date" id="dob" name="dob" required>
    <span id="dobError" class="error"></span><br>

    <label for="joiningDate">Joining Date:</label>
    <input type="date" id="joiningDate" name="joiningDate" required>
    <span id="joiningDateError" class="error"></span><br>

    <label for="salary">Salary:</label>
    <input type="number" id="salary" name="salary" required>
    <span id="salaryError" class="error"></span><br>

    <input type="submit" value="Submit">
</form>

<script>
    function validateForm() {
        // Get form inputs
        var dob = new Date(document.getElementById('dob').value);
        var joiningDate = new Date(document.getElementById('joiningDate').value);
        var salary = parseFloat(document.getElementById('salary').value);

        // Validate Date of Birth
        if (isNaN(dob.getTime())) {
            document.getElementById('dobError').innerHTML = 'Invalid Date of Birth';
            return false;
        } else {
            document.getElementById('dobError').innerHTML = '';
        }

        // Validate Joining Date
        if (isNaN(joiningDate.getTime())) {
            document.getElementById('joiningDateError').innerHTML = 'Invalid Joining Date';
            return false;
        } else {
            document.getElementById('joiningDateError').innerHTML = '';
        }

        // Validate Salary (positive number)
        if (isNaN(salary) || salary <= 0) {
            document.getElementById('salaryError').innerHTML = 'Salary should be a positive number';
            return false;
        } else {
            document.getElementById('salaryError').innerHTML = '';
        }

        // If all validations pass, the form is valid
        return true;
    }
</script>

</body>
</html>

slip 3// Write a JAVA Program to implement built-in support (java.util.Observable) Weather
station with members temperature, humidity, pressure and methods
mesurmentsChanged(), setMesurment(), getTemperature(), getHumidity(),
getPressure(



import java.util.Observable;
import java.util.Observer;

class WeatherData extends Observable {
    private float temperature;
    private float humidity;
    private float pressure;

    public void measurementsChanged() {
        setChanged();
        notifyObservers();
    }

    public void setMeasurements(float temperature, float humidity, float pressure) {
        this.temperature = temperature;
        this.humidity = humidity;
        this.pressure = pressure;
        measurementsChanged();
    }

    public float getTemperature() {
        return temperature;
    }

    public float getHumidity() {
        return humidity;
    }

    public float getPressure() {
        return pressure;
    }
}

class Display implements Observer {
    private WeatherData weatherData;

    public Display(WeatherData weatherData) {
        this.weatherData = weatherData;
        weatherData.addObserver(this);
    }

    @Override
    public void update(Observable o, Object arg) {
        if (o instanceof WeatherData) {
            WeatherData weatherData = (WeatherData) o;
            display(weatherData.getTemperature(), weatherData.getHumidity(), weatherData.getPressure());
        }
    }

    public void display(float temperature, float humidity, float pressure) {
        System.out.println("Temperature: " + temperature + "°C");
        System.out.println("Humidity: " + humidity + "%");
        System.out.println("Pressure: " + pressure + " hPa");
    }
}

public class Question3 {
    public static void main(String[] args) {
        WeatherData weatherData = new WeatherData();

        Display display = new Display(weatherData);

        // Simulate weather data changes
        weatherData.setMeasurements(25.0f, 60.0f, 1013.2f);
    }
}







// Write a python program to make Categorical values in numeric format for a given 
dataset

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Create a sample DataFrame (replace this with your dataset)
data = {'Category': ['A', 'B', 'C', 'A', 'B', 'C'],
        'Value': [10, 20, 30, 40, 50, 60]}

df = pd.DataFrame(data)

# Step 1: Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Step 2: Apply Label Encoding to the 'Category' column
df['Category_encoded'] = label_encoder.fit_transform(df['Category'])

# Display the DataFrame with encoded values
print("Original DataFrame:")
print(df)

# Step 3: Inverse Transform (if needed)
df['Category_decoded'] = label_encoder.inverse_transform(df['Category_encoded'])

# Display the DataFrame with the original and decoded values
print("\nDataFrame with decoded values:")
print(df)






// Create an HTML form for Login and write a JavaScript to validate email ID 
using Regular Expression. 


<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login Form</title>
    <style>
        .error {
            color: red;
        }
    </style>
</head>
<body>

<h2>Login Form</h2>

<form id="loginForm" onsubmit="return validateForm()">
    <label for="email">Email:</label>
    <input type="text" id="email" name="email" required>
    <span id="emailError" class="error"></span><br>

    <label for="password">Password:</label>
    <input type="password" id="password" name="password" required><br>

    <input type="submit" value="Login">
</form>

<script>
    function validateForm() {
        // Get form inputs
        var email = document.getElementById('email').value;

        // Regular expression for a simple email validation
        var emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

        // Validate email using regular expression
        if (!emailRegex.test(email)) {
            document.getElementById('emailError').innerHTML = 'Invalid email format';
            return false;
        } else {
            document.getElementById('emailError').innerHTML = '';
        }

        // If email validation passes, the form is valid
        return true;
    }
</script>

</body>
</html>


slip 4// Write a Java Program to implement Factory method for Pizza Store with createPizza(),
orederPizza(), prepare(), Bake(), cut(), box(). Use this to create variety of pizza’s like
NyStyleCheesePizza, ChicagoStyleCheesePizza etc


import java.util.ArrayList;
import java.util.List;

// Product: Pizza
abstract class Pizza {
    String name;

    abstract void prepare();

    void bake() {
        System.out.println("Baking " + name + " pizza");
    }

    void cut() {
        System.out.println("Cutting " + name + " pizza");
    }

    void box() {
        System.out.println("Boxing " + name + " pizza");
    }
}

// Concrete Product: NYStyleCheesePizza
class NYStyleCheesePizza extends Pizza {
    NYStyleCheesePizza() {
        name = "NY Style Cheese Pizza";
    }

    @Override
    void prepare() {
        System.out.println("Preparing ingredients for NY Style Cheese Pizza");
    }
}

// Concrete Product: ChicagoStyleCheesePizza
class ChicagoStyleCheesePizza extends Pizza {
    ChicagoStyleCheesePizza() {
        name = "Chicago Style Cheese Pizza";
    }

    @Override
    void prepare() {
        System.out.println("Preparing ingredients for Chicago Style Cheese Pizza");
    }
}

// Creator: PizzaStore
abstract class PizzaStore {
    public Pizza orderPizza(String type) {
        Pizza pizza = createPizza(type);

        pizza.prepare();
        pizza.bake();
        pizza.cut();
        pizza.box();

        return pizza;
    }

    protected abstract Pizza createPizza(String type);
}

// Concrete Creator: NYPizzaStore
class NYPizzaStore extends PizzaStore {
    @Override
    protected Pizza createPizza(String type) {
        if (type.equalsIgnoreCase("cheese")) {
            return new NYStyleCheesePizza();
        }
        // Add more pizza types here...
        return null;
    }
}

// Concrete Creator: ChicagoPizzaStore
class ChicagoPizzaStore extends PizzaStore {
    @Override
    protected Pizza createPizza(String type) {
        if (type.equalsIgnoreCase("cheese")) {
            return new ChicagoStyleCheesePizza();
        }
        // Add more pizza types here...
        return null;
    }
}

public class Question4 {
    public static void main(String[] args) {
        PizzaStore nyStore = new NYPizzaStore();
        PizzaStore chicagoStore = new ChicagoPizzaStore();

        Pizza pizza1 = nyStore.orderPizza("cheese");
        System.out.println("Ethan ordered a " + pizza1.name + "\n");

        Pizza pizza2 = chicagoStore.orderPizza("cheese");
        System.out.println("Joel ordered a " + pizza2.name + "\n");
    }
}










// Write a python program to Implement Simple Linear Regression for predicting house 
price.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data (replace this with your dataset)
house_size = np.array([1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700])
house_price = np.array([245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000])

# Reshape data
house_size = house_size.reshape(-1, 1)

# Step 1: Create a Linear Regression model
model = LinearRegression()

# Step 2: Fit the model to the data
model.fit(house_size, house_price)

# Step 3: Make predictions
house_size_to_predict = np.array([1500, 1800, 2000]).reshape(-1, 1)
predicted_prices = model.predict(house_size_to_predict)

# Step 4: Plot the data and the regression line
plt.scatter(house_size, house_price, color='blue', label='Actual Prices')
plt.plot(house_size, model.predict(house_size), color='red', label='Regression Line')
plt.xlabel('House Size (sqft)')
plt.ylabel('Price ($)')
plt.title('House Price Prediction')
plt.legend()
plt.show()

# Step 5: Display the predictions
print("Predicted Prices for House Sizes:")
for i in range(len(house_size_to_predict)):
 print(f"House Size: {house_size_to_predict[i][0]} sqft, Predicted Price: ${predicted_prices[i]:.2f}")












// Create a Node.js file that will convert the output "Hello World!" into upper-case letters.

// convertToUpper.js

const message = "Hello World!";
const upperCaseMessage = message.toUpperCase();

console.log(upperCaseMessage);


slip 5// Write a Java Program to implement Adapter pattern for Enumeration iterator

import java.util.Enumeration;
import java.util.Iterator;

// Target interface (Iterator)
interface IteratorAdapter<T> extends Iterator<T> {
}

// Adaptee interface (Enumeration)
class EnumerationAdapter<T> implements Enumeration<T> {
    private IteratorAdapter<T> iterator;

    public EnumerationAdapter(IteratorAdapter<T> iterator) {
        this.iterator = iterator;
    }

    @Override
    public boolean hasMoreElements() {
        return iterator.hasNext();
    }

    @Override
    public T nextElement() {
        return iterator.next();
    }
}

// Client class that uses Iterator interface
class Client {
    public void printValues(Iterator<String> iterator) {
        while (iterator.hasNext()) {
            System.out.println(iterator.next());
        }
    }
}

public class AdapterPatternExample {
    public static void main(String[] args) {
        // Create an Enumeration
        Enumeration<String> enumeration = new VectorAdapter<>(new Vector<String>().elements());

        // Adapt Enumeration to Iterator using the Adapter
        IteratorAdapter<String> iteratorAdapter = new EnumerationToIteratorAdapter<>(enumeration);

        // Client code can now use Iterator interface
        Client client = new Client();
        client.printValues(iteratorAdapter);
    }
}











// Write a python program to implement Multiple Linear Regression for given dataset.


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset (replace this with your dataset)
data = {
    'Size': [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700],
    'Bedrooms': [3, 3, 2, 3, 2, 3, 4, 4, 3, 2],
    'Location': [1, 2, 1, 3, 2, 1, 3, 2, 1, 2],
    'Price': [245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000]
}

df = pd.DataFrame(data)

# Step 1: Split the dataset into independent variables (features) and the dependent variable (target)
X = df[['Size', 'Bedrooms', 'Location']]
y = df['Price']

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create a Linear Regression model and fit it to the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Make predictions on the test data
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Step 6: Make predictions for new data
new_data = np.array([[1500, 3, 2]])  # New house features: Size, Bedrooms, Location
predicted_price = model.predict(new_data)

print("Predicted Price for a New House:", predicted_price[0])
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Sample dataset (replace this with your dataset)
data = {
    'Size': [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700],
    'Bedrooms': [3, 3, 2, 3, 2, 3, 4, 4, 3, 2],
    'Location': [1, 2, 1, 3, 2, 1, 3, 2, 1, 2],
    'Price': [245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000]
}

df = pd.DataFrame(data)

# Step 1: Split the dataset into independent variables (features) and the dependent variable (target)
X = df[['Size', 'Bedrooms', 'Location']]
y = df['Price']

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create a Random Forest Regressor model and fit it to the training data
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Make predictions on the test data
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)
print("Mean Absolute Error:", mae)

# Step 6: Make predictions for a new house (example: Size=1500, Bedrooms=3, Location=2)
new_data = np.array([[1500, 3, 2]])  # New house features: Size, Bedrooms, Location
predicted_price = model.predict(new_data)

print("Predicted Price for a New House:", predicted_price[0])














// Using nodejs create a web page to read two file names from user and append contents 
of first file into second file.

const express = require('express');
const bodyParser = require('body-parser');
const fs = require('fs');

const app = express();
const port = 3000;

app.use(bodyParser.urlencoded({ extended: true }));

app.get('/', (req, res) => {
    res.sendFile(__dirname + '/index.html');
});

app.post('/appendFiles', (req, res) => {
    const { firstFileName, secondFileName } = req.body;

    // Read the contents of the first file
    fs.readFile(firstFileName, 'utf8', (err, data) => {
        if (err) {
            return res.send('Error reading the first file.');
        }

        // Append the contents to the second file
        fs.appendFile(secondFileName, data, (err) => {
            if (err) {
                return res.send('Error appending to the second file.');
            }

            res.send('Contents appended successfully!');
        });
    });
});

app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});
Create a file named index.html in the same directory with the following content:
html
Copy code
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>File Appender</title>
</head>
<body>
    <h1>File Appender</h1>
    <form action="/appendFiles" method="post">
        <label for="firstFileName">Enter the name of the first file:</label>
        <input type="text" id="firstFileName" name="firstFileName" required>
        <br>
        <label for="secondFileName">Enter the name of the second file:</label>
        <input type="text" id="secondFileName" name="secondFileName" required>
        <br>
        <button type="submit">Append Files</button>
    </form>
</body>
</html>


slip 6// Write a Java Program to implement command pattern to test Remote Control

interface Command {
public void execute();
}
class Light {
public void on(){
System.out.println("Light is on");
}
public void off()
{
System.out.println("Light is off");
}
}
class LightOnCommand implements Command {
Light l1;
public LightOnCommand(Light a) {
this.l1 = a;
}
public void execute() {
l1.on();
}
}
class LightOffCommand implements Command {
Light l1;
public LightOffCommand(Light a) {
this.l1 = a;
}
public void execute() {
l1.off();
}
}
class SimpleRemoteControl {
Command slot;
public SimpleRemoteControl() {}
public void setCommand(Command command) {
slot = command;
}
public void buttonWasPressed() {
slot.execute();
}
}
public class Question5 {
public static void main(String[] args) {
SimpleRemoteControl r1 = new SimpleRemoteControl();
Light l1 = new Light();
LightOnCommand lo = new LightOnCommand(l1);
r1.setCommand(lo);
r1.buttonWasPressed();
LightOffCommand lO = new LightOffCommand(l1);
r1.setCommand(lO);
r1.buttonWasPressed();
}
}





// Write a python program to implement Polynomial Linear Regression for given dataset


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Sample dataset (replace this with your dataset)
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 8, 18, 32, 50])

# Reshape the data
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

# Step 1: Create PolynomialFeatures to transform the features
poly = PolynomialFeatures(degree=2)  # You can adjust the degree as needed

# Transform the features to include polynomial terms
x_poly = poly.fit_transform(x)

# Step 2: Create a Linear Regression model
model = LinearRegression()

# Step 3: Fit the model to the transformed features
model.fit(x_poly, y)

# Step 4: Predict new values using the polynomial model
x_new = np.array([[6]])  # Value to predict for
x_new_poly = poly.transform(x_new)
y_new = model.predict(x_new_poly)

# Step 5: Plot the data and the polynomial regression line
plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(x, model.predict(x_poly), color='red', label='Polynomial Regression Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polynomial Linear Regression')
plt.legend()
plt.show()

# Step 6: Display the predicted value
print(f"Predicted value for x={x_new[0][0]} is y={y_new[0][0]:.2f}")





//Create a Node.js file that opens the requested file and returns the content to the client. If anything goes wrong, throw a 404 error
Here's an example using Express:

1. Install the necessary module:

```bash
npm init -y
npm install express
```

2. Create a file named `fileServer.js`:

```javascript
// fileServer.js
const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
const port = 3000;

app.get('/getFile/:filename', (req, res) => {
  const requestedFile = req.params.filename;
  const filePath = path.join(__dirname, requestedFile);

  fs.readFile(filePath, 'utf8', (err, data) => {
    if (err) {
      if (err.code === 'ENOENT') {
        // File not found
        res.status(404).send('File not found');
      } else {
        // Other error
        res.status(500).send('Internal Server Error');
      }
    } else {
      // Send file content to the client
      res.send(data);
    }
  });
});

app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}/`);
});
```

3. Run the server:

```bash
node fileServer.js
```

4. Now you can access files by visiting URLs like http://localhost:3000/getFile/yourfilename.txt in your browser or using tools like curl or Postman.

Note: This example uses Express for simplicity. In a production environment, you may want to add more error handling, security measures, and consider serving static files differently.

