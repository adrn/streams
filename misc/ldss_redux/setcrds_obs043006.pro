pro setcrds_obs043006, strct

;  This program sets the structure cards for April 30, 2006
;  Magellan LDSS3 run


  print, 'Setting the cards for obs043006 data'
  nimg = n_elements(strct)


; BIAS FRAMES
;  strct[fnd_indx(strct,125):fnd_indx(strct,135)].type = 'ZRO';
;  strct[fnd_indx(strct,125):fnd_indx(strct,135)].flg_anly =1
;  strct[fnd_indx(strct,125):fnd_indx(strct,135)].bias_fil = 'Bias/bias'

; AFTERNOON CALIBRATIONS
  strct[fnd_indx(strct,136):fnd_indx(strct,151)].flg_anly =0
 



; OBJECTS -- NIGHT 2
  strct[fnd_indx(strct,152):fnd_indx(strct,164)].objname   = '162130' ; 78
  strct[fnd_indx(strct,165):fnd_indx(strct,174)].objname   = '262647' ; 79 *** use hbeta
  strct[fnd_indx(strct,177):fnd_indx(strct,184)].objname   = '327197' ; 80
  strct[fnd_indx(strct,185):fnd_indx(strct,192)].objname   = '172859' ; 81
  strct[fnd_indx(strct,193):fnd_indx(strct,200)].objname   = '6716'   ; 82
  strct[fnd_indx(strct,203):fnd_indx(strct,210)].objname   = '371747' ; 83
  strct[fnd_indx(strct,211):fnd_indx(strct,218)].objname   = '178719' ; 84  
  strct[fnd_indx(strct,219):fnd_indx(strct,228)].objname   = '307723' ; 85
  strct[fnd_indx(strct,229):fnd_indx(strct,237)].objname   = '154344' ; 86
  strct[fnd_indx(strct,238):fnd_indx(strct,245)].objname   = '1076'   ; 87





;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 162130

  objnum=78
  strct[fnd_indx(strct,152):fnd_indx(strct,164)].objnum  = objnum
  strct[fnd_indx(strct,159):fnd_indx(strct,164)].flg_anly= 1

  strct[fnd_indx(strct,159):fnd_indx(strct,161)].type    = 'OBJ'
  strct[fnd_indx(strct,162)].type = 'ARC'
  strct[fnd_indx(strct,163):fnd_indx(strct,164)].type    = 'FLT'

  strct[fnd_indx(strct,155)].flg_anly= 1
  strct[fnd_indx(strct,155)].type = 'IMG'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 262647  *** Hbeta
 
  objnum=79
  strct[fnd_indx(strct,165):fnd_indx(strct,176)].objnum  = objnum
  strct[fnd_indx(strct,170):fnd_indx(strct,176)].flg_anly= 1

  strct[fnd_indx(strct,171):fnd_indx(strct,173)].type    = 'OBJ'
  strct[fnd_indx(strct,175):fnd_indx(strct,176)].type    = 'FLT'
  strct[fnd_indx(strct,174)].type = 'ARC'

  strct[fnd_indx(strct,166)].flg_anly= 1
  strct[fnd_indx(strct,166)].type = 'IMG'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 327197

  objnum=80
  strct[fnd_indx(strct,177):fnd_indx(strct,184)].objnum  = objnum
  strct[fnd_indx(strct,178):fnd_indx(strct,184)].flg_anly= 1

  strct[fnd_indx(strct,179):fnd_indx(strct,181)].type    = 'OBJ'
  strct[fnd_indx(strct,183):fnd_indx(strct,184)].type    = 'FLT'
  strct[fnd_indx(strct,182)].type = 'ARC'
  strct[fnd_indx(strct,178)].type = 'IMG'

;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 172859

  objnum=81
  strct[fnd_indx(strct,185):fnd_indx(strct,192)].objnum  = objnum
  strct[fnd_indx(strct,186):fnd_indx(strct,192)].flg_anly= 1

  strct[fnd_indx(strct,187):fnd_indx(strct,189)].type    = 'OBJ'
  strct[fnd_indx(strct,191):fnd_indx(strct,192)].type    = 'FLT'
  strct[fnd_indx(strct,190)].type = 'ARC'
  strct[fnd_indx(strct,186)].type = 'IMG'

;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 6716

  objnum=82
  strct[fnd_indx(strct,193):fnd_indx(strct,200)].objnum  = objnum
  strct[fnd_indx(strct,194):fnd_indx(strct,200)].flg_anly= 1

  strct[fnd_indx(strct,195):fnd_indx(strct,197)].type    = 'OBJ'
  strct[fnd_indx(strct,199):fnd_indx(strct,200)].type    = 'FLT'
  strct[fnd_indx(strct,198)].type = 'ARC'
  strct[fnd_indx(strct,194)].type = 'IMG'

;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 371747

  objnum=83
  strct[fnd_indx(strct,203):fnd_indx(strct,210)].objnum  = objnum
  strct[fnd_indx(strct,204):fnd_indx(strct,210)].flg_anly= 1

  strct[fnd_indx(strct,205):fnd_indx(strct,207)].type    = 'OBJ'
  strct[fnd_indx(strct,209):fnd_indx(strct,210)].type    = 'FLT'
  strct[fnd_indx(strct,208)].type = 'ARC'
  strct[fnd_indx(strct,204)].type = 'IMG'

;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 178719

  objnum=84
  strct[fnd_indx(strct,211):fnd_indx(strct,218)].objnum  = objnum
  strct[fnd_indx(strct,212):fnd_indx(strct,218)].flg_anly= 1

  strct[fnd_indx(strct,213):fnd_indx(strct,215)].type    = 'OBJ'
  strct[fnd_indx(strct,217):fnd_indx(strct,218)].type    = 'FLT'
  strct[fnd_indx(strct,216)].type = 'ARC'
  strct[fnd_indx(strct,212)].type = 'IMG'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 307723

  objnum=85
  strct[fnd_indx(strct,219):fnd_indx(strct,228)].objnum  = objnum
  strct[fnd_indx(strct,220):fnd_indx(strct,228)].flg_anly= 1

  strct[fnd_indx(strct,221):fnd_indx(strct,225)].type    = 'OBJ'
  strct[fnd_indx(strct,227):fnd_indx(strct,228)].type    = 'FLT'
  strct[fnd_indx(strct,226)].type = 'ARC'
  strct[fnd_indx(strct,220)].type = 'IMG'

;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 154344

  objnum=86
  strct[fnd_indx(strct,229):fnd_indx(strct,237)].objnum  = objnum
  strct[fnd_indx(strct,230):fnd_indx(strct,237)].flg_anly= 1

  strct[fnd_indx(strct,232):fnd_indx(strct,234)].type    = 'OBJ'
  strct[fnd_indx(strct,236):fnd_indx(strct,237)].type    = 'FLT'
  strct[fnd_indx(strct,235)].type = 'ARC'
  strct[fnd_indx(strct,230)].type = 'IMG'

;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 1076

  objnum=87
  strct[fnd_indx(strct,238):fnd_indx(strct,245)].objnum  = objnum
  strct[fnd_indx(strct,239):fnd_indx(strct,245)].flg_anly= 1

  strct[fnd_indx(strct,240):fnd_indx(strct,242)].type    = 'OBJ'
  strct[fnd_indx(strct,244):fnd_indx(strct,245)].type    = 'FLT'
  strct[fnd_indx(strct,243)].type = 'ARC'
  strct[fnd_indx(strct,239)].type = 'IMG'


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  print, 'Setcrds finished.'


return
end
