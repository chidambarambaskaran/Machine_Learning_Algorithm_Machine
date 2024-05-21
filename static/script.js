function toggleTheme() {
    const body = document.body;
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle.checked) {
        body.classList.add('dark');
        body.classList.remove('light');
    } else {
        body.classList.add('light');
        body.classList.remove('dark');
    }
}

// Ensure the theme matches the toggle on page load
document.addEventListener('DOMContentLoaded', (event) => {
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle.checked) {
        document.body.classList.add('dark');
        document.body.classList.remove('light');
    } else {
        document.body.classList.add('light');
        document.body.classList.remove('dark');
    }
});


function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];

    const formData = new FormData();
    formData.append('file', file);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
        if (data.success) {
            window.uploadedData = data.df;
            document.getElementById('outputArea').innerText = data.message;
        } else {
            document.getElementById('outputArea').innerText = data.error;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('outputArea').innerText = 'Error uploading file. Please try again.';
    });
}

function evaluateClassification(algorithm) {
    fetch('/evaluate_classification', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ algorithm: algorithm, data: window.uploadedData })
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
        displayResult(data, 'classification-result');
    })
    .catch(error => console.error('Error:', error));
}

function evaluateRegression(algorithm) {
    fetch('/evaluate_regression', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ algorithm: algorithm, data: window.uploadedData })
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
        displayResult(data, 'regression-result');
    })
    .catch(error => console.error('Error:', error));
}

function displayResult(data, elementId) {
    const resultArea = document.getElementById(elementId);
    if (data.accuracy !== undefined) {
        resultArea.innerText = 'Accuracy: ' + data.accuracy.toFixed(2);
    } else if (data.mse !== undefined) {
        resultArea.innerText = 'Accuracy: ' + data.mse.toFixed(2);
    } else {
        resultArea.innerText = JSON.stringify(data, null, 2);
    }
}
