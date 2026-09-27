param(
    [Parameter(Mandatory = $true)][string]$Path,
    [double]$CX = 97.17, [double]$CY = 97.17,
    [double[]]$Angles = @(0, 45, 90, 135, 180, 225, 270, 315),
    [double]$RMax = 100, [double]$RStep = 0.05
)

Add-Type -AssemblyName System.Drawing
$bmp = New-Object System.Drawing.Bitmap $Path
$w = $bmp.Width; $h = $bmp.Height
$rect = New-Object System.Drawing.Rectangle 0, 0, $w, $h
$bd = $bmp.LockBits($rect, [System.Drawing.Imaging.ImageLockMode]::ReadOnly, [System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
$stride = $bd.Stride
$bytes = New-Object byte[] ($stride * $h)
[System.Runtime.InteropServices.Marshal]::Copy($bd.Scan0, $bytes, 0, $bytes.Length)
$bmp.UnlockBits($bd)
$bmp.Dispose()

# bilinear luminance at continuous coords (pixel centre of index i is i+0.5)
function LumAt([double]$fx, [double]$fy) {
    $x = $fx - 0.5; $y = $fy - 0.5
    $x0 = [math]::Floor($x); $y0 = [math]::Floor($y)
    $tx = $x - $x0; $ty = $y - $y0
    $acc = 0.0
    for ($dy = 0; $dy -le 1; $dy++) {
        for ($dx = 0; $dx -le 1; $dx++) {
            $xi = [math]::Min([math]::Max([int]($x0 + $dx), 0), $w - 1)
            $yi = [math]::Min([math]::Max([int]($y0 + $dy), 0), $h - 1)
            $i = $yi * $stride + $xi * 3
            $l = 0.114 * $bytes[$i] + 0.587 * $bytes[$i + 1] + 0.299 * $bytes[$i + 2]
            $wgt = (1 - [math]::Abs($dx - $tx)) * (1 - [math]::Abs($dy - $ty))
            $acc += $l * $wgt
        }
    }
    return $acc
}

foreach ($deg in $Angles) {
    $rad = $deg * [math]::PI / 180.0
    # image y grows downward; use -sin so angles read as standard math angles
    $ux = [math]::Cos($rad); $uy = -[math]::Sin($rad)

    $crossings = @()
    $prevR = 0.0
    $prevL = LumAt $CX $CY
    for ($r = $RStep; $r -le $RMax; $r += $RStep) {
        $l = LumAt ($CX + $ux * $r) ($CY + $uy * $r)
        if (($prevL - 127.5) * ($l - 127.5) -lt 0) {
            $t = (127.5 - $prevL) / ($l - $prevL)
            $pos = $prevR + $t * ($r - $prevR)
            $dir = if ($l -lt $prevL) { "->ink " } else { "->pap " }
            $crossings += ("{0}{1:N2}" -f $dir, $pos)
        }
        $prevR = $r; $prevL = $l
    }
    Write-Output ("angle {0,4}  {1}" -f $deg, ($crossings -join "  "))
}
