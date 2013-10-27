pro ldssstr__define

;  This routine defines the structure for spectral single-slit
; reductions of LDSS3 data

  tmp = {ldssstr, $
         ccdframe: ' ',   $  ; CCD FRAME NAME (in rawdata)
         ccdnum: 0L,      $  ; CCD FRAME Number
         flg_anly: 0,     $  ; Analysis flag 0=Don't Analyse
         type: ' ',       $  ; ObjTyp: OBJ, STD, DRK, ZRO, FLT, SKY, ARC
         objname: ' ',    $  ; Object Name
         objnum: 0L,      $  ; Object Num
         exp: 0.d,        $  ; Exposure time
         dateobs:' ',     $  ; Date of Obs
         UT: ' ',         $  ; UT
         RA: 0.0,         $  ; RA-D
         DEC: 0.0,        $  ; DEC-D
         SRA: ' ',        $  ; RA - string
         SDEC: ' ',       $  ; DEC- string
         Equinox: 0.,     $  ; EQUINOX 
         airmass:   0.,   $  ; Airmass
         rotang: 0.,      $  ; ROTATION ANGLE
         grism: ' ',      $  ; LDSS grism
         slitwidth:0.0,   $  ; slit size in arcseconds
         chip:intarr(4),  $  ; chip number 
         gain:fltarr(4),  $  ; Gain
         readno:fltarr(4),$  ; Read Noise
         bias_fil: ' '    $  ; bias file name
         }

end
  
         
