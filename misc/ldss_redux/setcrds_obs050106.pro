pro setcrds_obs050106, strct

;  This program sets the structure cards for May 1, 2006
;  Magellan LDSS3 run


  print, 'Setting the cards for obs050106 data'
  nimg = n_elements(strct)


; BIAS FRAMES
;  strct[fnd_indx(strct,125):fnd_indx(strct,135)].type = 'ZRO';
;  strct[fnd_indx(strct,125):fnd_indx(strct,135)].flg_anly =1
;  strct[fnd_indx(strct,125):fnd_indx(strct,135)].bias_fil = 'Bias/bias'

; AFTERNOON CALIBRATIONS
  strct[fnd_indx(strct,1030):fnd_indx(strct,1073)].flg_anly =0
 



; OBJECTS -- NIGHT 2
  strct[fnd_indx(strct,1074):fnd_indx(strct,1082)].objname   = '189157' ; 88




;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 189157

  objnum=88
  strct[fnd_indx(strct,1074):fnd_indx(strct,1082)].objnum  = objnum
  strct[fnd_indx(strct,1075):fnd_indx(strct,1082)].flg_anly= 1

  strct[fnd_indx(strct,1076):fnd_indx(strct,1079)].type    = 'OBJ'
  strct[fnd_indx(strct,1080)].type = 'ARC'
  strct[fnd_indx(strct,1081):fnd_indx(strct,1082)].type    = 'FLT'
  strct[fnd_indx(strct,1075)].type = 'IMG'

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  print, 'Setcrds finished.'


return
end
