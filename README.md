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
