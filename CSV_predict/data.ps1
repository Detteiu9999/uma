param(
    [Parameter(Mandatory = $true)]
    [string]$FolderPath
)

$ErrorActionPreference = "Stop"

# S = 19th column = zero-based index 18
# AF = 32nd column = zero-based index 31
$StartIndex = 18
$EndIndex = 31

function ConvertTo-CsvField {
    param(
        [AllowNull()]
        [string]$Value
    )

    if ($null -eq $Value) {
        return ""
    }

    if ($Value -match '[,"\r\n]') {
        return '"' + $Value.Replace('"', '""') + '"'
    }

    return $Value
}

if (-not (Test-Path -LiteralPath $FolderPath -PathType Container)) {
    throw "Folder not found: $FolderPath"
}

# Required for TextFieldParser
Add-Type -AssemblyName Microsoft.VisualBasic

$csvFiles = Get-ChildItem -LiteralPath $FolderPath -Filter "*.csv" -File

foreach ($csvFile in $csvFiles) {
    $tempFile = $csvFile.FullName + ".tmp"
    $backupFile = $csvFile.FullName + ".bak"

    Write-Host "Processing: $($csvFile.Name)"

    # Do not overwrite an existing backup
    if (-not (Test-Path -LiteralPath $backupFile)) {
        Copy-Item -LiteralPath $csvFile.FullName -Destination $backupFile
        Write-Host "Backup created: $($csvFile.Name).bak"
    }

    $parser = $null
    $writer = $null

    try {
        $parser = New-Object Microsoft.VisualBasic.FileIO.TextFieldParser($csvFile.FullName)
        $parser.TextFieldType = [Microsoft.VisualBasic.FileIO.FieldType]::Delimited
        $parser.SetDelimiters(",")
        $parser.HasFieldsEnclosedInQuotes = $true
        $parser.TrimWhiteSpace = $false

        # UTF-8 with BOM
        $utf8Bom = New-Object System.Text.UTF8Encoding($true)
        $writer = New-Object System.IO.StreamWriter($tempFile, $false, $utf8Bom)

        while (-not $parser.EndOfData) {
            $fields = $parser.ReadFields()

            if ($null -eq $fields) {
                continue
            }

            $newFields = for ($i = 0; $i -lt $fields.Count; $i++) {
                if ($i -lt $StartIndex -or $i -gt $EndIndex) {
                    ConvertTo-CsvField -Value $fields[$i]
                }
            }

            $writer.WriteLine(($newFields -join ","))
        }

        $writer.Close()
        $writer = $null

        $parser.Close()
        $parser = $null

        # Replace only after successful completion
        Move-Item -LiteralPath $tempFile -Destination $csvFile.FullName -Force

        Write-Host "Done: $($csvFile.Name)"
    }
    catch {
        Write-Host "FAILED: $($csvFile.Name)" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red

        if (Test-Path -LiteralPath $tempFile) {
            Remove-Item -LiteralPath $tempFile -Force
        }

        throw
    }
    finally {
        if ($null -ne $writer) {
            $writer.Close()
        }

        if ($null -ne $parser) {
            $parser.Close()
        }
    }
}

Write-Host "Finished."