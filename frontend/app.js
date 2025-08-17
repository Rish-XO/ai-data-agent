// API Configuration
const API_URL = 'http://localhost:8000';

// Global variables
let selectedFile = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    setupDragAndDrop();
    setupFileInput();
});

// Tab Navigation
function showSection(section) {
    // Hide all sections
    document.querySelectorAll('.section').forEach(s => s.classList.add('hidden'));
    
    // Show selected section
    document.getElementById(`${section}-section`).classList.remove('hidden');
    
    // Update tab styles
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('bg-white', 'text-gray-800', 'font-medium');
        btn.classList.add('text-gray-600');
    });
    
    const activeTab = document.getElementById(`${section}-tab`);
    activeTab.classList.add('bg-white', 'text-gray-800', 'font-medium');
    activeTab.classList.remove('text-gray-600');
    
    // Load data if needed
    if (section === 'candidates') {
        loadCandidates();
    }
}

// Drag and Drop Setup
function setupDragAndDrop() {
    const dropZone = document.getElementById('drop-zone');
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('border-blue-500', 'bg-blue-50');
    });
    
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('border-blue-500', 'bg-blue-50');
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('border-blue-500', 'bg-blue-50');
        
        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type === 'application/pdf') {
            handleFileSelect(files[0]);
        } else {
            alert('Please upload a PDF file');
        }
    });
}

// File Input Setup
function setupFileInput() {
    const fileInput = document.getElementById('file-input');
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
}

// Handle File Selection
function handleFileSelect(file) {
    if (file.type !== 'application/pdf') {
        alert('Please select a PDF file');
        return;
    }
    
    selectedFile = file;
    document.getElementById('file-name').textContent = file.name;
    document.getElementById('file-selected').classList.remove('hidden');
    document.getElementById('upload-btn').disabled = false;
    document.getElementById('results').innerHTML = `
        <div class="text-center py-8">
            <i class="fas fa-file-check text-4xl text-green-500 mb-4"></i>
            <p class="text-gray-600">Ready to parse: ${file.name}</p>
        </div>
    `;
}

// Upload Resume
async function uploadResume() {
    if (!selectedFile) {
        alert('Please select a file first');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    // Show loading
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('upload-btn').disabled = true;
    document.getElementById('results').innerHTML = '';
    
    try {
        // Set longer timeout for AI processing
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 120000); // 2 minutes
        
        const response = await fetch(`${API_URL}/upload-resume`, {
            method: 'POST',
            body: formData,
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        displayResults(data.candidate);
        
        // Show success message
        showNotification('Resume parsed successfully!', 'success');
        
    } catch (error) {
        console.error('Upload error:', error);
        document.getElementById('results').innerHTML = `
            <div class="bg-red-50 border border-red-200 rounded-lg p-4">
                <p class="text-red-700">
                    <i class="fas fa-exclamation-circle mr-2"></i>
                    Error: ${error.message}
                </p>
            </div>
        `;
        showNotification('Failed to parse resume', 'error');
    } finally {
        document.getElementById('loading').classList.add('hidden');
        document.getElementById('upload-btn').disabled = false;
    }
}

// Display Results
function displayResults(candidate) {
    const resultsHtml = `
        <div class="space-y-4">
            <div class="bg-green-50 border border-green-200 rounded-lg p-3 mb-4">
                <p class="text-green-700 text-sm">
                    <i class="fas fa-check-circle mr-2"></i>
                    AI successfully extracted information
                </p>
            </div>
            
            <div class="border-b pb-3">
                <label class="text-sm text-gray-500">Name</label>
                <p class="font-semibold text-gray-800">${candidate.name || 'Not found'}</p>
            </div>
            
            <div class="border-b pb-3">
                <label class="text-sm text-gray-500">Email</label>
                <p class="font-semibold text-gray-800">
                    ${candidate.email ? `<i class="fas fa-envelope mr-2"></i>${candidate.email}` : 'Not found'}
                </p>
            </div>
            
            <div class="border-b pb-3">
                <label class="text-sm text-gray-500">Phone</label>
                <p class="font-semibold text-gray-800">
                    ${candidate.phone ? `<i class="fas fa-phone mr-2"></i>${candidate.phone}` : 'Not found'}
                </p>
            </div>
            
            <div class="border-b pb-3">
                <label class="text-sm text-gray-500">Experience</label>
                <p class="font-semibold text-gray-800">${candidate.experience_years} years</p>
            </div>
            
            <div class="border-b pb-3">
                <label class="text-sm text-gray-500">Current Role</label>
                <p class="font-semibold text-gray-800">${candidate.current_role || 'Not specified'}</p>
            </div>
            
            <div>
                <label class="text-sm text-gray-500">Skills (${candidate.skills.length})</label>
                <div class="flex flex-wrap gap-2 mt-2">
                    ${candidate.skills.map(skill => 
                        `<span class="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm">${skill}</span>`
                    ).join('')}
                    ${candidate.skills.length === 0 ? '<span class="text-gray-400">No skills detected</span>' : ''}
                </div>
            </div>
        </div>
    `;
    
    document.getElementById('results').innerHTML = resultsHtml;
}

// Load All Candidates
async function loadCandidates() {
    document.getElementById('candidates-loading').classList.remove('hidden');
    document.getElementById('candidates-list').innerHTML = '';
    
    try {
        const response = await fetch(`${API_URL}/candidates`);
        const candidates = await response.json();
        
        if (candidates.length === 0) {
            document.getElementById('candidates-list').innerHTML = `
                <p class="text-gray-500 text-center py-8">No candidates found. Upload some resumes first!</p>
            `;
        } else {
            displayCandidatesTable(candidates);
        }
    } catch (error) {
        console.error('Error loading candidates:', error);
        document.getElementById('candidates-list').innerHTML = `
            <div class="bg-red-50 border border-red-200 rounded-lg p-4">
                <p class="text-red-700">
                    <i class="fas fa-exclamation-circle mr-2"></i>
                    Error loading candidates: ${error.message}
                </p>
            </div>
        `;
    } finally {
        document.getElementById('candidates-loading').classList.add('hidden');
    }
}

// Display Candidates Table
function displayCandidatesTable(candidates) {
    const tableHtml = `
        <table class="min-w-full">
            <thead class="bg-gray-50">
                <tr>
                    <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Name</th>
                    <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Email</th>
                    <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Experience</th>
                    <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Skills</th>
                    <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Added</th>
                </tr>
            </thead>
            <tbody class="bg-white divide-y divide-gray-200">
                ${candidates.map(candidate => `
                    <tr class="hover:bg-gray-50">
                        <td class="px-4 py-3 text-sm font-medium text-gray-900">${candidate.name}</td>
                        <td class="px-4 py-3 text-sm text-gray-500">${candidate.email || '-'}</td>
                        <td class="px-4 py-3 text-sm text-gray-500">${candidate.experience_years} years</td>
                        <td class="px-4 py-3 text-sm text-gray-500">
                            <div class="flex flex-wrap gap-1">
                                ${JSON.parse(candidate.skills || '[]').slice(0, 3).map(skill => 
                                    `<span class="px-2 py-1 bg-blue-100 text-blue-700 rounded text-xs">${skill}</span>`
                                ).join('')}
                                ${JSON.parse(candidate.skills || '[]').length > 3 ? 
                                    `<span class="text-gray-400 text-xs">+${JSON.parse(candidate.skills).length - 3} more</span>` : ''}
                            </div>
                        </td>
                        <td class="px-4 py-3 text-sm text-gray-500">
                            ${new Date(candidate.created_at).toLocaleDateString()}
                        </td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
    
    document.getElementById('candidates-list').innerHTML = tableHtml;
}

// Search by Skill
async function searchBySkill() {
    const skill = document.getElementById('skill-search').value.trim();
    
    if (!skill) {
        alert('Please enter a skill to search');
        return;
    }
    
    document.getElementById('search-results').innerHTML = `
        <div class="text-center py-8">
            <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
            <p class="mt-3 text-gray-600">Searching for "${skill}"...</p>
        </div>
    `;
    
    try {
        const response = await fetch(`${API_URL}/search?skill=${encodeURIComponent(skill)}`);
        const candidates = await response.json();
        
        if (candidates.length === 0) {
            document.getElementById('search-results').innerHTML = `
                <div class="text-center py-8">
                    <i class="fas fa-search text-6xl text-gray-300 mb-4"></i>
                    <p class="text-gray-500">No candidates found with skill: ${skill}</p>
                </div>
            `;
        } else {
            document.getElementById('search-results').innerHTML = `
                <p class="text-sm text-gray-600 mb-4">Found ${candidates.length} candidate(s) with "${skill}"</p>
                <div id="search-results-table"></div>
            `;
            
            // Create a temporary div to hold the table
            const tempDiv = document.createElement('div');
            tempDiv.id = 'candidates-list';
            document.getElementById('search-results-table').appendChild(tempDiv);
            
            displayCandidatesTable(candidates);
            
            // Move the table to search results
            document.getElementById('search-results-table').innerHTML = tempDiv.innerHTML;
        }
    } catch (error) {
        console.error('Search error:', error);
        document.getElementById('search-results').innerHTML = `
            <div class="bg-red-50 border border-red-200 rounded-lg p-4">
                <p class="text-red-700">
                    <i class="fas fa-exclamation-circle mr-2"></i>
                    Error searching: ${error.message}
                </p>
            </div>
        `;
    }
}

// Show Notification
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `fixed top-4 right-4 px-6 py-3 rounded-lg shadow-lg z-50 ${
        type === 'success' ? 'bg-green-500 text-white' : 
        type === 'error' ? 'bg-red-500 text-white' : 
        'bg-blue-500 text-white'
    }`;
    notification.innerHTML = `
        <i class="fas ${
            type === 'success' ? 'fa-check-circle' : 
            type === 'error' ? 'fa-exclamation-circle' : 
            'fa-info-circle'
        } mr-2"></i>
        ${message}
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, 3000);
}