const form = document.getElementById('uploadForm');
const fileInput = document.getElementById('csvFile');
const outputDiv = document.getElementById('output');
const spinner = document.getElementById('loadingSpinner');

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  const file = fileInput.files[0];
  if (!file) return alert('Please upload a CSV file.');

  const formData = new FormData();
  formData.append('file', file);

  spinner.style.display = 'block';
  outputDiv.innerHTML = '';

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData
    });

    const data = await response.json();
    spinner.style.display = 'none';

    if (response.ok) {
      outputDiv.innerHTML = `
        <h5>Results:</h5>
        <ul class="list-group">
          <li class="list-group-item">Accuracy: ${data.accuracy.toFixed(4)}</li>
          <li class="list-group-item">Precision: ${data.precision.toFixed(4)}</li>
          <li class="list-group-item">Recall: ${data.recall.toFixed(4)}</li>
          <li class="list-group-item">F1 Score: ${data.f1.toFixed(4)}</li>
          <li class="list-group-item">ROC AUC: ${data.roc_auc.toFixed(4)}</li>
        </ul>
        <img src="data:image/png;base64,${data.confusion_matrix_plot}" class="img-fluid mt-3" alt="Confusion Matrix">
      `;
    } else {
      outputDiv.innerHTML = `<div class="alert alert-danger">${data.error || 'Something went wrong.'}</div>`;
    }
  } catch (error) {
    spinner.style.display = 'none';
    outputDiv.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
  }
});
