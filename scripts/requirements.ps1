# PowerShell script to install packages from requirements.txt

$REQUIREMENTS_FILE = "requirements.txt"

# Check if requirements.txt exists
if (-not (Test-Path $REQUIREMENTS_FILE)) {
    Write-Host "Error: $REQUIREMENTS_FILE not found in the current directory." -ForegroundColor Red
    exit 1
}

# Read requirements file, ignoring comments and empty lines
Write-Host "`nReading requirements from $REQUIREMENTS_FILE...`n"
$failed_packages = @()

# Read file content and process each line
Get-Content $REQUIREMENTS_FILE | ForEach-Object {
    $package = $_.Trim()
    
    # Skip empty lines and comments
    if ([string]::IsNullOrWhiteSpace($package) -or $package.StartsWith("#")) {
        return
    }

    Write-Host "Installing package: $package..."

    # Attempt to install the package
    python -m pip install $package
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Successfully installed: $package" -ForegroundColor Green
    } else {
        Write-Host "Failed to install: $package" -ForegroundColor Red
        $failed_packages += $package
    }
    Write-Host
}

# Display a summary of failed packages
if ($failed_packages.Count -ne 0) {
    Write-Host "`nThe following packages failed to install:" -ForegroundColor Red
    foreach ($pkg in $failed_packages) {
        Write-Host "- $pkg" -ForegroundColor Red
    }
    exit 1
} else {
    Write-Host "`nAll packages installed successfully!" -ForegroundColor Green
    exit 0
}