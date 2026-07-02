// AutoML Platform - Main JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const progressContainer = document.getElementById('progress-container');
    const progressText = document.getElementById('progress-text');

    if (uploadForm) {
        uploadForm.addEventListener('submit', function() {
            if (progressContainer) {
                progressContainer.classList.remove('d-none');
            }
            if (progressText) {
                progressText.textContent = 'Uploading and analyzing your data...';
            }
        });
    }
});
