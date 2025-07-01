if ($env:ENABLE_PROFILING -eq "1") {
    Write-Host "Disabling line profiling..."
    Remove-Item Env:ENABLE_PROFILING
} else {
    Write-Host "Enabling line profiling..."
    $env:ENABLE_PROFILING = "1"
}

Write-Host "`nCurrent value of ENABLE_PROFILING: $env:ENABLE_PROFILING"