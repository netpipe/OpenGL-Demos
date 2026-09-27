param(
    [Parameter(Mandatory = $true)][string[]]$Images,
    [Parameter(Mandatory = $true)][string[]]$Labels,
    [Parameter(Mandatory = $true)][string]$Out,
    [int]$Scale = 3,
    [int]$Pad = 14
)

Add-Type -AssemblyName System.Drawing

$bmps = @()
foreach ($p in $Images) { $bmps += (New-Object System.Drawing.Bitmap (Resolve-Path -LiteralPath $p).Path) }

$labelH = 30
$w = 0; $h = 0
foreach ($b in $bmps) {
    $w += $b.Width * $Scale + $Pad
    $bh = $b.Height * $Scale
    if ($bh -gt $h) { $h = $bh }
}
$w += $Pad

$canvas = New-Object System.Drawing.Bitmap $w, ($h + $labelH + $Pad * 2)
$g = [System.Drawing.Graphics]::FromImage($canvas)
$g.Clear([System.Drawing.Color]::FromArgb(245, 245, 247))
$g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::NearestNeighbor
$g.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::Half

$font = New-Object System.Drawing.Font "Segoe UI", 13, ([System.Drawing.FontStyle]::Bold)
$brush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(30, 30, 35))
$pen = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(200, 200, 205)), 1

$x = $Pad
for ($i = 0; $i -lt $bmps.Count; $i++) {
    $bw = $bmps[$i].Width * $Scale
    $bh = $bmps[$i].Height * $Scale
    $g.DrawString($Labels[$i], $font, $brush, $x, $Pad)
    $rect = New-Object System.Drawing.Rectangle $x, ($Pad + $labelH), $bw, $bh
    $g.DrawImage($bmps[$i], $rect)
    $g.DrawRectangle($pen, $rect)
    $x += $bw + $Pad
}

$g.Dispose()
$canvas.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
$canvas.Dispose()
foreach ($b in $bmps) { $b.Dispose() }
Write-Output "wrote $Out"
