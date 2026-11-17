# Build the report .docx (and a PDF preview) from report_src.md with Microsoft Word (COM).
param([string]$src, [string]$root, [string]$outDocx, [string]$outPdf)
$ErrorActionPreference = "Stop"
$FONT = "Angsana New"
$script:PH1 = "[ตำแหน่งรูป]"; $script:PH2 = "ที่มา:"; $script:REFHEAD = "เอกสารอ้างอิง"
$lines = [IO.File]::ReadAllLines($src, [Text.Encoding]::UTF8)

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$doc = $word.Documents.Add()
try {
    $ps = $doc.PageSetup
    $ps.PaperSize = 7          # A4
    $ps.TopMargin = 108; $ps.LeftMargin = 108; $ps.BottomMargin = 72; $ps.RightMargin = 72

    function SetFont($st, $size, $bold) {
        $st.Font.Name = $FONT; $st.Font.NameBi = $FONT; $st.Font.NameAscii = $FONT; $st.Font.NameOther = $FONT
        $st.Font.Size = $size; $st.Font.SizeBi = $size
        $st.Font.Bold = [int]$bold; $st.Font.BoldBi = [int]$bold
        $st.Font.Italic = 0; $st.Font.ItalicBi = 0
        $st.Font.Color = -16777216
    }
    function PStyle($name) {
        try { return $doc.Styles.Item($name) } catch { $s = $doc.Styles.Add($name, 1); $s.BaseStyle = $doc.Styles.Item(-1); return $s }
    }
    $normal = $doc.Styles.Item(-1); SetFont $normal 16 $false
    $normal.ParagraphFormat.SpaceBefore = 0; $normal.ParagraphFormat.SpaceAfter = 0; $normal.ParagraphFormat.LineSpacingRule = 0
    $normal.NoSpaceBetweenParagraphsOfSameStyle = $false
    $h1 = $doc.Styles.Item(-2); SetFont $h1 20 $true
    $h1.ParagraphFormat.Alignment = 1; $h1.ParagraphFormat.PageBreakBefore = -1; $h1.ParagraphFormat.SpaceBefore = 0; $h1.ParagraphFormat.SpaceAfter = 18; $h1.ParagraphFormat.KeepWithNext = -1
    $h2 = $doc.Styles.Item(-3); SetFont $h2 18 $true
    $h2.ParagraphFormat.SpaceBefore = 12; $h2.ParagraphFormat.SpaceAfter = 4; $h2.ParagraphFormat.KeepWithNext = -1; $h2.ParagraphFormat.Alignment = 0
    $h3 = $doc.Styles.Item(-4); SetFont $h3 16 $true
    $h3.ParagraphFormat.SpaceBefore = 8; $h3.ParagraphFormat.SpaceAfter = 2; $h3.ParagraphFormat.KeepWithNext = -1; $h3.ParagraphFormat.Alignment = 0
    foreach ($id in -20, -21, -22) { $t = $doc.Styles.Item($id); SetFont $t 16 $false }

    $body = PStyle "ReportBody"; SetFont $body 16 $false
    $body.ParagraphFormat.FirstLineIndent = 36; $body.ParagraphFormat.Alignment = 9; $body.ParagraphFormat.SpaceAfter = 0
    $front = PStyle "FrontHead"; SetFont $front 20 $true
    $front.ParagraphFormat.Alignment = 1; $front.ParagraphFormat.PageBreakBefore = -1; $front.ParagraphFormat.SpaceAfter = 18; $front.ParagraphFormat.KeepWithNext = -1
    $title = PStyle "TitleBig"; SetFont $title 24 $true; $title.ParagraphFormat.Alignment = 1; $title.ParagraphFormat.SpaceAfter = 6
    $center = PStyle "CenterText"; SetFont $center 16 $false; $center.ParagraphFormat.Alignment = 1
    $figp = PStyle "FigPara"; SetFont $figp 16 $false; $figp.ParagraphFormat.Alignment = 1; $figp.ParagraphFormat.KeepWithNext = -1; $figp.ParagraphFormat.SpaceBefore = 8
    $figc = PStyle "FigCaption"; SetFont $figc 16 $true; $figc.ParagraphFormat.Alignment = 1; $figc.ParagraphFormat.SpaceAfter = 10
    $tabc = PStyle "TabCaption"; SetFont $tabc 16 $true; $tabc.ParagraphFormat.Alignment = 0; $tabc.ParagraphFormat.KeepWithNext = -1; $tabc.ParagraphFormat.SpaceBefore = 8
    $tabt = PStyle "TableText"; SetFont $tabt 14 $false; $tabt.ParagraphFormat.Alignment = 0
    $eqp = PStyle "EqPara"; SetFont $eqp 16 $false; $eqp.ParagraphFormat.Alignment = 1; $eqp.ParagraphFormat.SpaceBefore = 4; $eqp.ParagraphFormat.SpaceAfter = 4
    $note = PStyle "NoteBox"; SetFont $note 15 $false; $note.Font.Italic = 1; $note.Font.ItalicBi = 1
    $note.ParagraphFormat.LeftIndent = 18; $note.ParagraphFormat.RightIndent = 18; $note.ParagraphFormat.SpaceBefore = 4; $note.ParagraphFormat.SpaceAfter = 4
    $note.ParagraphFormat.Shading.BackgroundPatternColor = 10092543
    $ph = PStyle "FigPlaceholder"; SetFont $ph 15 $false; $ph.ParagraphFormat.Alignment = 1; $ph.ParagraphFormat.KeepWithNext = -1
    $ph.ParagraphFormat.Shading.BackgroundPatternColor = 15132390; $ph.ParagraphFormat.SpaceBefore = 8
    $ph.ParagraphFormat.Borders.Enable = 1
    $code = PStyle "CodeBlock"; SetFont $code 10 $false; $code.Font.Name = "Consolas"; $code.Font.NameAscii = "Consolas"
    $code.ParagraphFormat.Shading.BackgroundPatternColor = 15987699; $code.ParagraphFormat.LeftIndent = 18
    $num = PStyle "NumItem"; SetFont $num 16 $false; $num.ParagraphFormat.LeftIndent = 36; $num.ParagraphFormat.FirstLineIndent = -18
    $bul = $doc.Styles.Item(-49); SetFont $bul 16 $false
    $ref = PStyle "RefItem"; SetFont $ref 16 $false; $ref.ParagraphFormat.LeftIndent = 36; $ref.ParagraphFormat.FirstLineIndent = -36; $ref.ParagraphFormat.SpaceAfter = 4

    $sel = $word.Selection
    $script:frontMode = ""

    function TypeInline([string]$text) {
        $parts = [regex]::Split($text, '(\*\*.+?\*\*|`[^`]+`)')
        foreach ($p in $parts) {
            if ($p -eq '') { continue }
            if ($p.StartsWith('**') -and $p.EndsWith('**') -and $p.Length -ge 4) {
                $sel.Font.Bold = 1; $sel.Font.BoldBi = 1; $sel.TypeText($p.Substring(2, $p.Length - 4)); $sel.Font.Bold = 0; $sel.Font.BoldBi = 0
            } elseif ($p.StartsWith('`') -and $p.EndsWith('`') -and $p.Length -ge 2) {
                $old = $sel.Font.Name; $sel.Font.Name = "Consolas"; $sel.Font.Size = 11; $sel.TypeText($p.Substring(1, $p.Length - 2)); $sel.Font.Name = $old; $sel.Font.Size = 16
            } else { $sel.TypeText($p) }
        }
    }
    function Para($style, [string]$text) {
        $sel.Style = $style
        $sel.Font.Bold = 0; $sel.Font.BoldBi = 0
        TypeInline $text
        $sel.TypeParagraph()
    }
    function CellText([string]$t) { return ($t -replace '\*\*', '' -replace '`', '').Trim() }

    function AddTable($rows) {
        $data = @()
        foreach ($r in $rows) {
            $cells = $r.Trim().Trim('|').Split('|') | ForEach-Object { CellText $_ }
            if (($cells -join '') -match '^[\s\-:]+$') { continue }
            $data += , $cells
        }
        $nr = $data.Count; $nc = ($data | ForEach-Object { $_.Count } | Measure-Object -Maximum).Maximum
        $sel.Style = $tabt
        $t = $doc.Tables.Add($sel.Range, $nr, $nc)
        $t.Borders.Enable = 1
        $t.Range.Style = $tabt
        for ($i = 0; $i -lt $nr; $i++) {
            for ($j = 0; $j -lt $nc; $j++) {
                $v = if ($j -lt $data[$i].Count) { $data[$i][$j] } else { "" }
                $t.Cell($i + 1, $j + 1).Range.Text = $v
            }
        }
        $hdr = $t.Rows.Item(1)
        $hdr.Range.Font.Bold = 1; $hdr.Range.Font.BoldBi = 1
        $hdr.Shading.BackgroundPatternColor = 14277081
        $hdr.HeadingFormat = -1
        $t.AutoFitBehavior(1)
        $t.AutoFitBehavior(2)
        $t.Rows.AllowBreakAcrossPages = 0
        $sel.EndKey(6) | Out-Null
        $sel.Style = $normal
        $sel.TypeParagraph()
    }

    function AddFigure([string]$spec) {
        $f = $spec.Split('|')
        $rel = $f[0].Trim(); $cap = $f[1].Trim(); $srcNote = if ($f.Count -gt 2) { $f[2].Trim() } else { "" }
        $path = Join-Path $root ($rel -replace '/', '\')
        if (Test-Path $path) {
            $sel.Style = $figp
            $shape = $sel.InlineShapes.AddPicture($path, $false, $true)
            $shape.LockAspectRatio = -1
            if ($shape.Width -gt 410) { $shape.Width = 410 }
            if ($shape.Height -gt 560) { $shape.Height = 560 }
            $sel.EndKey(6) | Out-Null
            $sel.TypeParagraph()
        } else {
            $sel.Style = $ph
            $sel.TypeText([string]$script:PH1 + " " + $rel)
            $sel.TypeText([char]11)
            $sel.TypeText([string]$script:PH2 + " " + $srcNote)
            $sel.TypeParagraph()
        }
        Para $figc $cap
    }

    function AddEq([string]$lin) {
        $sel.Style = $eqp
        $start = $sel.Start
        $sel.TypeText($lin)
        $r = $doc.Range($start, $sel.End)
        try { $m = $doc.OMaths.Add($r); $m.OMaths.Item(1).BuildUp() } catch { }
        $sel.EndKey(6) | Out-Null
        $sel.TypeParagraph()
    }

    $tbuf = New-Object System.Collections.Generic.List[string]
    $inCode = $false
    for ($k = 0; $k -lt $lines.Count; $k++) {
        $L = $lines[$k]
        if ($inCode) {
            if ($L -eq '@endcode') { $inCode = $false; continue }
            Para $code $L; continue
        }
        if ($L.StartsWith('|')) { $tbuf.Add($L); if ($k + 1 -ge $lines.Count -or -not $lines[$k + 1].StartsWith('|')) { AddTable $tbuf; $tbuf.Clear() }; continue }
        if ($L.Trim() -eq '') { continue }
        if ($L -eq '@code') { $inCode = $true; continue }
        if ($L.StartsWith('@space ')) { $n = [int]$L.Substring(7); for ($i = 0; $i -lt $n; $i++) { $sel.Style = $center; $sel.TypeParagraph() }; continue }
        if ($L.StartsWith('@title ')) { Para $title $L.Substring(7); continue }
        if ($L.StartsWith('@center ')) { Para $center $L.Substring(8); continue }
        if ($L.StartsWith('@fronthead ')) {
            $t = $L.Substring(11); $script:frontMode = $t
            Para $front $t; continue
        }
        if ($L -eq '@toc') { $sel.Style = $normal; $doc.TablesOfContents.Add($sel.Range, $true, 1, 3) | Out-Null; $sel.EndKey(6) | Out-Null; $sel.TypeParagraph(); continue }
        if ($L -eq '@lot' -or $L -eq '@lof') {
            $st = if ($L -eq '@lot') { 'TabCaption' } else { 'FigCaption' }
            $sel.Style = $normal
            $doc.Fields.Add($sel.Range, -1, ('TOC \h \z \t "' + $st + ',1"'), $false) | Out-Null
            $sel.EndKey(6) | Out-Null; $sel.TypeParagraph(); continue
        }
        if ($L.StartsWith('### ')) { Para $h3 $L.Substring(4); continue }
        if ($L.StartsWith('## ')) { Para $h2 $L.Substring(3); continue }
        if ($L.StartsWith('# ')) { $script:frontMode = $L.Substring(2); Para $h1 $L.Substring(2); continue }
        if ($L.StartsWith('@tcap ')) { Para $tabc $L.Substring(6); continue }
        if ($L.StartsWith('@fig ')) { AddFigure $L.Substring(5); continue }
        if ($L.StartsWith('@eq ')) { AddEq $L.Substring(4); continue }
        if ($L.StartsWith('> ')) { Para $note $L.Substring(2); continue }
        if ($L.StartsWith('- ')) { Para $bul $L.Substring(2); continue }
        if ($L -match '^\d+\)\s') { Para $num $L; continue }
        if ($script:frontMode -like '*Reference*' -or $script:frontMode -eq $script:REFHEAD) { Para $ref $L; continue }
        Para $body $L
    }

    # page numbers centred in the footer
    $doc.Sections.Item(1).Footers.Item(1).PageNumbers.Add(1, $true) | Out-Null
    $doc.TablesOfContents.Item(1).Update()
    $doc.Fields.Update() | Out-Null
    $doc.TablesOfContents.Item(1).Update()
    $doc.SaveAs2($outDocx, 16)
    $doc.ExportAsFixedFormat($outPdf, 17)
    "pages: " + $doc.ComputeStatistics(2)
    "tables: " + $doc.Tables.Count + ", pictures: " + $doc.InlineShapes.Count + ", equations: " + $doc.OMaths.Count
}
finally {
    $doc.Close(0)
    if ($word.Documents.Count -eq 0) { $word.Quit() }
    [Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
