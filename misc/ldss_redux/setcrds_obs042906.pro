pro setcrds_obs042906, strct

;  This program sets the structure cards for April 29, 2006
;  Magellan LDSS3 run


  print, 'Setting the cards for obs042906 data'
  nimg = n_elements(strct)


; BIAS FRAMES
  strct[fnd_indx(strct,125):fnd_indx(strct,135)].type = 'ZRO'
  strct[fnd_indx(strct,125):fnd_indx(strct,135)].flg_anly =1
  strct[fnd_indx(strct,125):fnd_indx(strct,135)].bias_fil = 'Bias/bias'

; AFTERNOON CALIBRATIONS
  strct[fnd_indx(strct,1):fnd_indx(strct,26)].flg_anly =0
  strct[fnd_indx(strct,115):fnd_indx(strct,124)].flg_anly =1



; OBJECTS -- NIGHT 1
  strct[fnd_indx(strct,27):fnd_indx(strct,37)].objname   = '268302' ; 70
  strct[fnd_indx(strct,38):fnd_indx(strct,45)].objname   = '266041' ; 71
  strct[fnd_indx(strct,46):fnd_indx(strct,53)].objname   = '67068'  ; 72
  strct[fnd_indx(strct,71):fnd_indx(strct,78)].objname   = '370242' ; 73
  strct[fnd_indx(strct,79):fnd_indx(strct,87)].objname   = '301586' ; 74
  strct[fnd_indx(strct,89):fnd_indx(strct,96)].objname   = '11400'  ; 75
  strct[fnd_indx(strct,97):fnd_indx(strct,106)].objname  = '188703' ; 76  
  strct[fnd_indx(strct,107):fnd_indx(strct,114)].objname = '205988' ; 77





;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 268302

  objnum=70
  strct[fnd_indx(strct,27):fnd_indx(strct,37)].objnum  = objnum
  strct[fnd_indx(strct,31):fnd_indx(strct,37)].flg_anly= 1

  strct[fnd_indx(strct,32):fnd_indx(strct,34)].type    = 'OBJ'
  strct[fnd_indx(strct,35)].type = 'ARC'
  strct[fnd_indx(strct,36):fnd_indx(strct,37)].type    = 'FLT'

;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 266041

  objnum=71
  strct[fnd_indx(strct,38):fnd_indx(strct,45)].objnum  = objnum
  strct[fnd_indx(strct,40):fnd_indx(strct,45)].flg_anly= 1

  strct[fnd_indx(strct,42):fnd_indx(strct,42)].type    = 'OBJ'
  strct[fnd_indx(strct,44):fnd_indx(strct,45)].type    = 'FLT'
  strct[fnd_indx(strct,43)].type = 'ARC'

;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 67068

  objnum=72
  strct[fnd_indx(strct,46):fnd_indx(strct,53)].objnum  = objnum
  strct[fnd_indx(strct,47):fnd_indx(strct,53)].flg_anly= 1

  strct[fnd_indx(strct,47):fnd_indx(strct,49)].type    = 'OBJ'
  strct[fnd_indx(strct,51):fnd_indx(strct,52)].type    = 'FLT'
  strct[fnd_indx(strct,50)].type = 'ARC'
  strct[fnd_indx(strct,53)].type = 'IMG'

;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 370242

  objnum=73
  strct[fnd_indx(strct,71):fnd_indx(strct,78)].objnum  = objnum
  strct[fnd_indx(strct,72):fnd_indx(strct,78)].flg_anly= 1

  strct[fnd_indx(strct,73):fnd_indx(strct,75)].type    = 'OBJ'
  strct[fnd_indx(strct,77):fnd_indx(strct,78)].type    = 'FLT'
  strct[fnd_indx(strct,76)].type = 'ARC'
  strct[fnd_indx(strct,72)].type = 'IMG'

;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 301586

  objnum=74
  strct[fnd_indx(strct,79):fnd_indx(strct,87)].objnum  = objnum
  strct[fnd_indx(strct,80):fnd_indx(strct,87)].flg_anly= 1

  strct[fnd_indx(strct,81):fnd_indx(strct,84)].type    = 'OBJ'
  strct[fnd_indx(strct,86):fnd_indx(strct,87)].type    = 'FLT'
  strct[fnd_indx(strct,85)].type = 'ARC'
  strct[fnd_indx(strct,80)].type = 'IMG'

;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 11400

  objnum=75
  strct[fnd_indx(strct,88):fnd_indx(strct,96)].objnum  = objnum
  strct[fnd_indx(strct,89):fnd_indx(strct,96)].flg_anly= 1

  strct[fnd_indx(strct,90):fnd_indx(strct,93)].type    = 'OBJ'
  strct[fnd_indx(strct,95):fnd_indx(strct,96)].type    = 'FLT'
  strct[fnd_indx(strct,94)].type = 'ARC'
  strct[fnd_indx(strct,89)].type = 'IMG'

;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 188703

  objnum=76
  strct[fnd_indx(strct,97):fnd_indx(strct,106)].objnum  = objnum
  strct[fnd_indx(strct,98):fnd_indx(strct,106)].flg_anly= 1

  strct[fnd_indx(strct,99):fnd_indx(strct,103)].type    = 'OBJ'
  strct[fnd_indx(strct,105):fnd_indx(strct,106)].type    = 'FLT'
  strct[fnd_indx(strct,104)].type = 'ARC'
  strct[fnd_indx(strct,98)].type = 'IMG'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 205988

  objnum=77
  strct[fnd_indx(strct,107):fnd_indx(strct,114)].objnum  = objnum
  strct[fnd_indx(strct,108):fnd_indx(strct,114)].flg_anly= 1

  strct[fnd_indx(strct,109):fnd_indx(strct,111)].type    = 'OBJ'
  strct[fnd_indx(strct,113):fnd_indx(strct,114)].type    = 'FLT'
  strct[fnd_indx(strct,112)].type = 'ARC'
  strct[fnd_indx(strct,108)].type = 'IMG'


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  print, 'Setcrds finished.'


return
end
