/**
 * Ansel Photo Album - Frontend Application
 */

// DOM Elements
const setupSection = document.getElementById('setup-section');
const mainSection = document.getElementById('main-section');
const credentialsForm = document.getElementById('credentials-form');
const authSection = document.getElementById('auth-section');
const dropboxCredentialsForm = document.getElementById('dropbox-credentials-form');
const appKeyInput = document.getElementById('app-key');
const appSecretInput = document.getElementById('app-secret');
const accountInfo = document.getElementById('account-info');
const yearSelect = document.getElementById('year-select');
const themeSelect = document.getElementById('theme-select');
const qualitySelect = document.getElementById('quality-select');
const manageThemesBtn = document.getElementById('manage-themes-btn');
const scanBtn = document.getElementById('scan-btn');
const syncBtn = document.getElementById('sync-btn');
const analyzeBtn = document.getElementById('analyze-btn');
const downloadBtn = document.getElementById('download-btn');
const progressContainer = document.getElementById('progress-container');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const photoGrid = document.getElementById('photo-grid');
const folderModal = document.getElementById('folder-modal');
const folderList = document.getElementById('folder-list');
const cancelScanBtn = document.getElementById('cancel-scan-btn');
const startScanBtn = document.getElementById('start-scan-btn');

// Theme panel elements
const themePanel = document.getElementById('theme-panel');
const selectionCount = document.getElementById('selection-count');
const clearSelectionBtn = document.getElementById('clear-selection-btn');
const themeChips = document.getElementById('theme-chips');

// Theme modal elements
const themeModal = document.getElementById('theme-modal');
const themeListEl = document.getElementById('theme-list');
const newThemeInput = document.getElementById('new-theme-input');
const addThemeBtn = document.getElementById('add-theme-btn');
const closeThemeModalBtn = document.getElementById('close-theme-modal-btn');

// State
let currentYear = null;
let currentTheme = null;
let currentQuality = null;
let syncPollInterval = null;
let analyzePollInterval = null;
let downloadPollInterval = null;
let selectedPhotos = new Set(); // Temporary UI selection (for theme assignment)
let checkedPhotos = new Set(); // Persistent checked state (for batch downloads)
let lastClickedIndex = null;
let themes = [];
let photoElements = [];

/**
 * Initialize the application
 */
async function init() {
    // Check authentication status
    const status = await fetch('/api/auth/status').then(r => r.json());

    if (status.authenticated) {
        showMainSection(status.account);
        await loadYears();
        await loadThemes();
    } else if (status.has_credentials) {
        showSetupSection(true);
    } else {
        showSetupSection(false);
    }

    // Set up event listeners
    setupEventListeners();
    setupKeyboardShortcuts();
}

/**
 * Show setup section
 */
function showSetupSection(hasCredentials) {
    setupSection.classList.remove('hidden');
    mainSection.classList.add('hidden');

    if (hasCredentials) {
        credentialsForm.classList.add('hidden');
        authSection.classList.remove('hidden');
    } else {
        credentialsForm.classList.remove('hidden');
        authSection.classList.add('hidden');
    }
}

/**
 * Show main section
 */
function showMainSection(account) {
    setupSection.classList.add('hidden');
    mainSection.classList.remove('hidden');

    if (account) {
        accountInfo.textContent = `Connected as ${account.name}`;
    }
}

/**
 * Set up event listeners
 */
function setupEventListeners() {
    // Credentials form
    dropboxCredentialsForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const appKey = appKeyInput.value.trim();
        const appSecret = appSecretInput.value.trim();

        if (!appKey || !appSecret) {
            alert('Please enter both App Key and App Secret');
            return;
        }

        await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                dropbox_app_key: appKey,
                dropbox_app_secret: appSecret
            })
        });

        showSetupSection(true);
    });

    // Year select
    yearSelect.addEventListener('change', async () => {
        currentYear = yearSelect.value;
        syncBtn.disabled = !currentYear;
        analyzeBtn.disabled = !currentYear;
        downloadBtn.disabled = !currentYear;
        clearSelection();

        if (currentYear) {
            await loadPhotos(currentYear, currentTheme, currentQuality);
        } else {
            photoGrid.innerHTML = '<p class="placeholder">Select a year to view photos</p>';
        }
    });

    // Theme select
    themeSelect.addEventListener('change', async () => {
        currentTheme = themeSelect.value || null;
        clearSelection();

        if (currentYear) {
            await loadPhotos(currentYear, currentTheme, currentQuality);
        }
    });

    // Quality select
    qualitySelect.addEventListener('change', async () => {
        currentQuality = qualitySelect.value || null;
        clearSelection();

        if (currentYear) {
            await loadPhotos(currentYear, currentTheme, currentQuality);
        }
    });

    // Analyze button
    analyzeBtn.addEventListener('click', async () => {
        if (!currentYear) return;

        analyzeBtn.disabled = true;
        progressContainer.classList.remove('hidden');

        try {
            await fetch(`/api/quality/analyze/${currentYear}`, { method: 'POST' });
            startAnalyzePollling();
        } catch (err) {
            alert(`Analysis failed: ${err.message}`);
            analyzeBtn.disabled = false;
            progressContainer.classList.add('hidden');
        }
    });

    // Download button
    downloadBtn.addEventListener('click', async () => {
        if (!currentYear) return;

        downloadBtn.disabled = true;
        progressContainer.classList.remove('hidden');

        try {
            await fetch(`/api/download/${currentYear}`, { method: 'POST' });
            startDownloadPolling();
        } catch (err) {
            alert(`Download failed: ${err.message}`);
            downloadBtn.disabled = false;
            progressContainer.classList.add('hidden');
        }
    });

    // Manage themes button
    manageThemesBtn.addEventListener('click', () => {
        openThemeModal();
    });

    // Theme panel clear button
    clearSelectionBtn.addEventListener('click', () => {
        clearSelection();
    });

    // Theme modal buttons
    addThemeBtn.addEventListener('click', async () => {
        const name = newThemeInput.value.trim();
        if (name) {
            await createTheme(name);
            newThemeInput.value = '';
        }
    });

    newThemeInput.addEventListener('keypress', async (e) => {
        if (e.key === 'Enter') {
            const name = newThemeInput.value.trim();
            if (name) {
                await createTheme(name);
                newThemeInput.value = '';
            }
        }
    });

    closeThemeModalBtn.addEventListener('click', () => {
        themeModal.classList.add('hidden');
    });

    // Scan button - show folder selection modal
    scanBtn.addEventListener('click', async () => {
        folderModal.classList.remove('hidden');
        folderList.innerHTML = '<p>Loading folders...</p>';

        try {
            const data = await fetch('/api/folders').then(r => r.json());

            if (data.error) {
                folderList.innerHTML = `<p>Error: ${data.error}</p>`;
                return;
            }

            if (data.folders.length === 0) {
                folderList.innerHTML = '<p>No folders found in Dropbox</p>';
                return;
            }

            folderList.innerHTML = data.folders.map(folder => `
                <div class="folder-item">
                    <input type="checkbox" id="folder-${folder.name}" value="${folder.path}">
                    <label for="folder-${folder.name}">${folder.name}</label>
                </div>
            `).join('');

            // Pre-select common photo folders
            const commonFolders = ['Camera Uploads', 'Photos', 'Pictures'];
            for (const name of commonFolders) {
                const checkbox = document.getElementById(`folder-${name}`);
                if (checkbox) checkbox.checked = true;
            }
        } catch (err) {
            folderList.innerHTML = `<p>Error loading folders: ${err.message}</p>`;
        }
    });

    // Cancel scan
    cancelScanBtn.addEventListener('click', () => {
        folderModal.classList.add('hidden');
    });

    // Start scan with selected folders
    startScanBtn.addEventListener('click', async () => {
        const checkboxes = folderList.querySelectorAll('input[type="checkbox"]:checked');
        const paths = Array.from(checkboxes).map(cb => cb.value);

        if (paths.length === 0) {
            alert('Please select at least one folder');
            return;
        }

        folderModal.classList.add('hidden');
        scanBtn.disabled = true;
        scanBtn.textContent = 'Scanning...';

        try {
            const result = await fetch('/api/scan', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ paths })
            }).then(r => r.json());

            if (result.error) {
                alert(`Scan failed: ${result.error}`);
            } else {
                alert(`Found ${result.photo_count} photos across ${result.year_count} years`);
                await loadYears();
            }
        } catch (err) {
            alert(`Scan failed: ${err.message}`);
        } finally {
            scanBtn.disabled = false;
            scanBtn.textContent = 'Scan Dropbox';
        }
    });

    // Sync button
    syncBtn.addEventListener('click', async () => {
        if (!currentYear) return;

        syncBtn.disabled = true;
        progressContainer.classList.remove('hidden');

        try {
            await fetch(`/api/sync/${currentYear}`, { method: 'POST' });
            startSyncPolling();
        } catch (err) {
            alert(`Sync failed: ${err.message}`);
            syncBtn.disabled = false;
            progressContainer.classList.add('hidden');
        }
    });
}

/**
 * Load years into select
 */
async function loadYears() {
    const data = await fetch('/api/years').then(r => r.json());

    // Create a map of year data from API
    const yearMap = {};
    for (const year of data.years) {
        yearMap[year.year] = year;
    }

    // Clear existing options (except first)
    while (yearSelect.options.length > 1) {
        yearSelect.remove(1);
    }

    // Always show years 2022-2026
    const years = [2026, 2025, 2024, 2023, 2022];
    for (const year of years) {
        const option = document.createElement('option');
        option.value = year;
        if (yearMap[year]) {
            option.textContent = `${year} (${yearMap[year].count} photos, ${yearMap[year].synced} synced)`;
        } else {
            option.textContent = `${year}`;
        }
        yearSelect.appendChild(option);
    }
}

/**
 * Load photos for a year
 */
async function loadPhotos(year, theme = null, quality = null) {
    photoGrid.innerHTML = '<p class="placeholder">Loading photos...</p>';
    photoElements = [];
    checkedPhotos.clear(); // Clear checked state when loading new photos

    let url = `/api/photos/${year}`;
    const params = new URLSearchParams();
    if (theme) params.set('theme', theme);
    if (quality) params.set('quality', quality);
    if (params.toString()) url += `?${params.toString()}`;

    const data = await fetch(url).then(r => r.json());

    if (data.photos.length === 0) {
        let message = `No photos found for ${year}`;
        if (theme) message += ` with theme "${theme}"`;
        if (quality === 'good') message += ' (Good Photos filter)';
        if (quality === 'burst_best') message += ' (Best of Bursts filter)';
        photoGrid.innerHTML = `<p class="placeholder">${message}</p>`;
        return;
    }

    photoGrid.innerHTML = '';

    data.photos.forEach((photo, index) => {
        const item = document.createElement('div');
        item.className = 'photo-item';
        item.dataset.photoId = photo.id;
        item.dataset.index = index;

        if (photo.has_thumbnail) {
            const img = document.createElement('img');
            img.src = `/api/thumbnail/${photo.id}`;
            img.alt = photo.name;
            img.loading = 'lazy';
            item.appendChild(img);
        } else {
            item.classList.add('loading');
            item.title = photo.name;
        }

        // Add theme badges
        const photoThemes = photo.themes || [];
        if (photoThemes.length > 0) {
            const badgeContainer = document.createElement('div');
            badgeContainer.className = 'theme-badges';
            photoThemes.forEach(t => {
                const badge = document.createElement('span');
                badge.className = 'theme-badge';
                badge.textContent = t;
                badgeContainer.appendChild(badge);
            });
            item.appendChild(badgeContainer);
        }

        // Add click handler for selection (but not on checkbox)
        item.addEventListener('click', (e) => {
            // Ignore clicks on checkbox
            if (!e.target.closest('.checkbox')) {
                handlePhotoClick(e, photo.id, index);
            }
        });

        // Add checkbox for checked state (persistent)
        const checkbox = document.createElement('div');
        checkbox.className = 'checkbox';
        checkbox.innerHTML = '';
        checkbox.title = 'Check for batch processing';
        checkbox.addEventListener('click', (e) => {
            e.stopPropagation();
            handleCheckboxClick(photo.id, index);
        });
        item.appendChild(checkbox);

        // Load checked state
        if (photo.checked) {
            item.classList.add('checked');
            checkedPhotos.add(photo.id);
            checkbox.innerHTML = '&#10003;';
        }

        photoGrid.appendChild(item);
        photoElements.push({ element: item, photo });
    });
}

/**
 * Start polling for sync progress
 */
function startSyncPolling() {
    if (syncPollInterval) {
        clearInterval(syncPollInterval);
    }

    syncPollInterval = setInterval(async () => {
        const status = await fetch('/api/sync/status').then(r => r.json());

        progressFill.style.width = `${status.percent}%`;
        progressText.textContent = `Syncing: ${status.completed}/${status.total} (${status.percent}%) - ${status.current_file}`;

        // Refresh photo grid periodically during sync
        if (status.completed % 10 === 0 && currentYear) {
            await loadPhotos(currentYear, currentTheme, currentQuality);
        }

        if (!status.is_running) {
            clearInterval(syncPollInterval);
            syncPollInterval = null;
            syncBtn.disabled = false;

            // Final refresh
            await loadPhotos(currentYear, currentTheme, currentQuality);
            await loadYears();

            // Hide progress after a delay
            setTimeout(() => {
                progressContainer.classList.add('hidden');
                progressFill.style.width = '0%';
            }, 2000);
        }
    }, 500);
}

/**
 * Start polling for quality analysis progress
 */
function startAnalyzePollling() {
    if (analyzePollInterval) {
        clearInterval(analyzePollInterval);
    }

    analyzePollInterval = setInterval(async () => {
        const data = await fetch(`/api/quality/status?year=${currentYear}`).then(r => r.json());
        const status = data.progress;

        progressFill.style.width = `${status.percent}%`;
        progressText.textContent = `Analyzing: ${status.completed}/${status.total} (${status.percent}%) - ${status.current_file}`;

        if (!status.is_running) {
            clearInterval(analyzePollInterval);
            analyzePollInterval = null;
            analyzeBtn.disabled = false;

            // Final refresh
            await loadPhotos(currentYear, currentTheme, currentQuality);

            // Show stats
            const stats = data.stats;
            progressText.textContent = `Analysis complete: ${stats.good_photos} good photos (${stats.percent_good}%), ${stats.burst_groups} burst groups`;

            // Hide progress after a delay
            setTimeout(() => {
                progressContainer.classList.add('hidden');
                progressFill.style.width = '0%';
            }, 3000);
        }
    }, 500);
}

/**
 * Start polling for download progress
 */
function startDownloadPolling() {
    if (downloadPollInterval) {
        clearInterval(downloadPollInterval);
    }

    downloadPollInterval = setInterval(async () => {
        const status = await fetch(`/api/download/status`).then(r => r.json());

        progressFill.style.width = `${status.percent}%`;
        progressText.textContent = `Downloading: ${status.completed}/${status.total} (${status.percent}%) - ${status.current_file}`;

        if (!status.is_running) {
            clearInterval(downloadPollInterval);
            downloadPollInterval = null;
            downloadBtn.disabled = false;

            // Show completion message
            if (status.error) {
                progressText.textContent = `Download failed: ${status.error}`;
            } else {
                progressText.textContent = `Download complete: ${status.total} photos saved to ${status.download_path}`;
            }

            // Hide progress after a delay
            setTimeout(() => {
                progressContainer.classList.add('hidden');
                progressFill.style.width = '0%';
            }, 5000);
        }
    }, 500);
}

/**
 * Load themes into select and panel
 */
async function loadThemes() {
    const data = await fetch('/api/themes').then(r => r.json());
    themes = data.themes;

    // Update theme select dropdown
    while (themeSelect.options.length > 1) {
        themeSelect.remove(1);
    }

    // Add "Unthemed" option
    const unthemedOption = document.createElement('option');
    unthemedOption.value = 'unthemed';
    unthemedOption.textContent = `Unthemed (${data.unthemed_count})`;
    themeSelect.appendChild(unthemedOption);

    // Add themes
    for (const theme of themes) {
        const option = document.createElement('option');
        option.value = theme.name;
        option.textContent = `${theme.name} (${theme.count})`;
        themeSelect.appendChild(option);
    }

    // Update theme panel chips
    updateThemeChips();
}

/**
 * Update theme chips in the assignment panel
 */
function updateThemeChips() {
    themeChips.innerHTML = '';

    themes.forEach((theme, index) => {
        const chip = document.createElement('button');
        chip.className = 'theme-chip';
        chip.dataset.theme = theme.name;

        // Add number hint for first 9 themes
        const shortcut = index < 9 ? `${index + 1}` : '';
        chip.innerHTML = shortcut ? `<span class="shortcut">${shortcut}</span>${theme.name}` : theme.name;

        chip.addEventListener('click', () => assignThemeToSelected(theme.name));
        themeChips.appendChild(chip);
    });
}

/**
 * Handle photo click for selection
 */
function handlePhotoClick(event, photoId, index) {
    event.preventDefault();

    if (event.shiftKey && lastClickedIndex !== null) {
        // Range selection
        const start = Math.min(lastClickedIndex, index);
        const end = Math.max(lastClickedIndex, index);

        for (let i = start; i <= end; i++) {
            const pe = photoElements[i];
            if (pe) {
                selectedPhotos.add(pe.photo.id);
                pe.element.classList.add('selected');
            }
        }
    } else if (event.ctrlKey || event.metaKey) {
        // Toggle selection
        if (selectedPhotos.has(photoId)) {
            selectedPhotos.delete(photoId);
            photoElements[index].element.classList.remove('selected');
        } else {
            selectedPhotos.add(photoId);
            photoElements[index].element.classList.add('selected');
        }
    } else {
        // Single click - toggle selection
        if (selectedPhotos.has(photoId)) {
            selectedPhotos.delete(photoId);
            photoElements[index].element.classList.remove('selected');
        } else {
            selectedPhotos.add(photoId);
            photoElements[index].element.classList.add('selected');
        }
    }

    lastClickedIndex = index;
    updateSelectionUI();
}

/**
 * Handle checkbox click for checked state (persistent)
 */
async function handleCheckboxClick(photoId, index) {
    const pe = photoElements[index];
    if (!pe) return;

    const isCurrentlyChecked = checkedPhotos.has(photoId);
    const newCheckedState = !isCurrentlyChecked;

    // Update UI immediately
    if (newCheckedState) {
        checkedPhotos.add(photoId);
        pe.element.classList.add('checked');
        pe.element.querySelector('.checkbox').innerHTML = '&#10003;';
    } else {
        checkedPhotos.delete(photoId);
        pe.element.classList.remove('checked');
        pe.element.querySelector('.checkbox').innerHTML = '';
    }

    // Persist to backend
    try {
        await fetch(`/api/photos/${photoId}/checked`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ checked: newCheckedState })
        });
    } catch (err) {
        console.error('Failed to update checked state:', err);
        // Revert UI on error
        if (newCheckedState) {
            checkedPhotos.delete(photoId);
            pe.element.classList.remove('checked');
            pe.element.querySelector('.checkbox').innerHTML = '';
        } else {
            checkedPhotos.add(photoId);
            pe.element.classList.add('checked');
            pe.element.querySelector('.checkbox').innerHTML = '&#10003;';
        }
    }
}

/**
 * Select a single photo by index, clearing any existing selection
 */
function selectSinglePhoto(index) {
    if (index < 0 || index >= photoElements.length) return;

    // Clear existing selection
    selectedPhotos.clear();
    document.querySelectorAll('.photo-item.selected').forEach(el => {
        el.classList.remove('selected');
    });

    const pe = photoElements[index];
    selectedPhotos.add(pe.photo.id);
    pe.element.classList.add('selected');
    lastClickedIndex = index;

    // Scroll into view if needed
    pe.element.scrollIntoView({ block: 'nearest', behavior: 'smooth' });

    updateSelectionUI();
}

/**
 * Clear photo selection
 */
function clearSelection() {
    selectedPhotos.clear();
    lastClickedIndex = null;

    document.querySelectorAll('.photo-item.selected').forEach(el => {
        el.classList.remove('selected');
    });

    updateSelectionUI();
}

/**
 * Update selection UI (count and panel visibility)
 */
function updateSelectionUI() {
    const count = selectedPhotos.size;
    selectionCount.textContent = `${count} photo${count !== 1 ? 's' : ''} selected`;

    if (count > 0) {
        themePanel.classList.remove('hidden');
    } else {
        themePanel.classList.add('hidden');
    }
}

/**
 * Update theme badges on a photo element in place
 */
function updatePhotoThemeBadges(pe, newThemes) {
    // Update in-memory data
    pe.photo.themes = newThemes;

    // Remove existing badge container
    const existing = pe.element.querySelector('.theme-badges');
    if (existing) existing.remove();

    // Add new badges if any
    if (newThemes.length > 0) {
        const badgeContainer = document.createElement('div');
        badgeContainer.className = 'theme-badges';
        newThemes.forEach(t => {
            const badge = document.createElement('span');
            badge.className = 'theme-badge';
            badge.textContent = t;
            badgeContainer.appendChild(badge);
        });
        pe.element.appendChild(badgeContainer);
    }
}

/**
 * Assign theme to selected photos
 */
async function assignThemeToSelected(themeName, advanceToNext = false) {
    if (selectedPhotos.size === 0) return;

    const nextIndex = advanceToNext ? (lastClickedIndex !== null ? lastClickedIndex + 1 : 0) : -1;
    const photoIds = Array.from(selectedPhotos);

    // Update DOM immediately
    for (const pe of photoElements) {
        if (selectedPhotos.has(pe.photo.id)) {
            updatePhotoThemeBadges(pe, [themeName]);
        }
    }

    // Advance or clear selection immediately
    if (advanceToNext && nextIndex >= 0 && nextIndex < photoElements.length) {
        selectSinglePhoto(nextIndex);
    } else {
        clearSelection();
    }

    // Persist to backend in background
    fetch('/api/photos/bulk/themes', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            photo_ids: photoIds,
            themes: [themeName],
            mode: 'set'
        })
    }).then(() => loadThemes());
}

/**
 * Clear themes from selected photos (mark as unthemed)
 */
async function clearThemesFromSelected(advanceToNext = false) {
    if (selectedPhotos.size === 0) return;

    const nextIndex = advanceToNext ? (lastClickedIndex !== null ? lastClickedIndex + 1 : 0) : -1;
    const photoIds = Array.from(selectedPhotos);

    // Update DOM immediately
    for (const pe of photoElements) {
        if (selectedPhotos.has(pe.photo.id)) {
            updatePhotoThemeBadges(pe, []);
        }
    }

    // Advance or clear selection immediately
    if (advanceToNext && nextIndex >= 0 && nextIndex < photoElements.length) {
        selectSinglePhoto(nextIndex);
    } else {
        clearSelection();
    }

    // Persist to backend in background
    fetch('/api/photos/bulk/themes', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            photo_ids: photoIds,
            themes: [],
            mode: 'set'
        })
    }).then(() => loadThemes());
}

/**
 * Open theme management modal
 */
async function openThemeModal() {
    themeModal.classList.remove('hidden');
    await refreshThemeList();
}

/**
 * Refresh theme list in modal
 */
async function refreshThemeList() {
    const data = await fetch('/api/themes').then(r => r.json());

    themeListEl.innerHTML = '';

    for (const theme of data.themes) {
        const item = document.createElement('div');
        item.className = 'theme-list-item';

        const nameSpan = document.createElement('span');
        nameSpan.className = 'theme-name';
        nameSpan.textContent = theme.name;

        const countSpan = document.createElement('span');
        countSpan.className = 'theme-count';
        countSpan.textContent = `(${theme.count})`;

        item.appendChild(nameSpan);
        item.appendChild(countSpan);

        // Only custom themes can be deleted
        if (!theme.is_default) {
            const deleteBtn = document.createElement('button');
            deleteBtn.className = 'btn btn-small btn-danger';
            deleteBtn.textContent = 'Delete';
            deleteBtn.addEventListener('click', () => deleteTheme(theme.name));
            item.appendChild(deleteBtn);
        } else {
            const badge = document.createElement('span');
            badge.className = 'default-badge';
            badge.textContent = 'default';
            item.appendChild(badge);
        }

        themeListEl.appendChild(item);
    }
}

/**
 * Create a new custom theme
 */
async function createTheme(name) {
    const result = await fetch('/api/themes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name })
    }).then(r => r.json());

    if (result.error) {
        alert(result.error);
    } else {
        await refreshThemeList();
        await loadThemes();
    }
}

/**
 * Delete a custom theme
 */
async function deleteTheme(name) {
    if (!confirm(`Delete theme "${name}"? Photos with this theme will not be affected.`)) {
        return;
    }

    const result = await fetch(`/api/themes/${encodeURIComponent(name)}`, {
        method: 'DELETE'
    }).then(r => r.json());

    if (result.error) {
        alert(result.error);
    } else {
        await refreshThemeList();
        await loadThemes();
    }
}

/**
 * Setup keyboard shortcuts
 */
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Ignore if typing in an input
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
            return;
        }

        // Escape - clear selection
        if (e.key === 'Escape') {
            clearSelection();
            return;
        }

        // Arrow keys - navigate between photos
        if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
            e.preventDefault();
            if (photoElements.length === 0) return;

            let nextIndex;
            if (lastClickedIndex === null) {
                nextIndex = e.key === 'ArrowRight' ? 0 : photoElements.length - 1;
            } else {
                nextIndex = e.key === 'ArrowRight' ? lastClickedIndex + 1 : lastClickedIndex - 1;
            }

            if (nextIndex >= 0 && nextIndex < photoElements.length) {
                selectSinglePhoto(nextIndex);
            }
            return;
        }

        // Spacebar - toggle checked state on selected photos
        if (e.key === ' ') {
            e.preventDefault();
            if (selectedPhotos.size === 0) return;

            for (let i = 0; i < photoElements.length; i++) {
                const pe = photoElements[i];
                if (selectedPhotos.has(pe.photo.id)) {
                    handleCheckboxClick(pe.photo.id, i);
                }
            }
            return;
        }

        // Only process theme shortcuts if photos are selected
        if (selectedPhotos.size === 0) return;

        // U - clear themes (unthemed) and advance
        if (e.key === 'u' || e.key === 'U') {
            e.preventDefault();
            clearThemesFromSelected(true);
            return;
        }

        // 1-9 - assign theme by number and advance to next photo
        const num = parseInt(e.key);
        if (num >= 1 && num <= 9 && num <= themes.length) {
            e.preventDefault();
            assignThemeToSelected(themes[num - 1].name, true);
        }
    });
}

// Initialize on load
document.addEventListener('DOMContentLoaded', init);
