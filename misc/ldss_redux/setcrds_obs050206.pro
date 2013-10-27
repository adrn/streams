pro setcrds_obs050206, strct

;  This program sets the structure cards for May 2, 2006
;  Magellan LDSS3 run


  print, 'Setting the cards for obs050206 data'
  nimg = n_elements(strct)


; BIAS FRAMES
;  strct[fnd_indx(strct,125):fnd_indx(strct,135)].type = 'ZRO';
;  strct[fnd_indx(strct,125):fnd_indx(strct,135)].flg_anly =1
;  strct[fnd_indx(strct,125):fnd_indx(strct,135)].bias_fil = 'Bias/bias'

; AFTERNOON CALIBRATIONS
  strct[fnd_indx(strct,1083):fnd_indx(strct,1094)].flg_anly =0
 



; OBJECTS -- NIGHT 4
  strct[fnd_indx(strct,1095):fnd_indx(strct,1103)].objname   = '161656' ; 89
  strct[fnd_indx(strct,1104):fnd_indx(strct,1111)].objname   = '276603' ; 90
  strct[fnd_indx(strct,1112):fnd_indx(strct,1119)].objname   = '243181' ; 91
  strct[fnd_indx(strct,1120):fnd_indx(strct,1128)].objname   = '280272' ; 92
  strct[fnd_indx(strct,1192):fnd_indx(strct,1199)].objname   = '280272' ; 93




;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 161656

  objnum=89
  strct[fnd_indx(strct,1095):fnd_indx(strct,1103)].objnum  = objnum
  strct[fnd_indx(strct,1096):fnd_indx(strct,1103)].flg_anly= 1

  strct[fnd_indx(strct,1098):fnd_indx(strct,1100)].type    = 'OBJ'
  strct[fnd_indx(strct,1101)].type = 'ARC'
  strct[fnd_indx(strct,1102):fnd_indx(strct,1103)].type    = 'FLT'
  strct[fnd_indx(strct,1096)].type = 'IMG'

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 276603

  objnum=90
  strct[fnd_indx(strct,1104):fnd_indx(strct,1111)].objnum  = objnum
  strct[fnd_indx(strct,1105):fnd_indx(strct,1111)].flg_anly= 1

  strct[fnd_indx(strct,1106):fnd_indx(strct,1108)].type    = 'OBJ'
  strct[fnd_indx(strct,1109)].type = 'ARC'
  strct[fnd_indx(strct,1110):fnd_indx(strct,1111)].type    = 'FLT'
  strct[fnd_indx(strct,1105)].type = 'IMG'

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 243181

  objnum=91
  strct[fnd_indx(strct,1112):fnd_indx(strct,1119)].objnum  = objnum
  strct[fnd_indx(strct,1113):fnd_indx(strct,1119)].flg_anly= 1

  strct[fnd_indx(strct,1114):fnd_indx(strct,1116)].type    = 'OBJ'
  strct[fnd_indx(strct,1117)].type = 'ARC'
  strct[fnd_indx(strct,1118):fnd_indx(strct,1119)].type    = 'FLT'
  strct[fnd_indx(strct,1113)].type = 'IMG'

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 280272

  objnum=92
  strct[fnd_indx(strct,1120):fnd_indx(strct,1128)].objnum  = objnum
  strct[fnd_indx(strct,1121):fnd_indx(strct,1128)].flg_anly= 1

  strct[fnd_indx(strct,1123):fnd_indx(strct,1125)].type    = 'OBJ'
  strct[fnd_indx(strct,1126)].type = 'ARC'
  strct[fnd_indx(strct,1127):fnd_indx(strct,1128)].type    = 'FLT'
  strct[fnd_indx(strct,1121)].type = 'IMG'

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 179291

  objnum=93
  strct[fnd_indx(strct,1192):fnd_indx(strct,1199)].objnum  = objnum
  strct[fnd_indx(strct,1193):fnd_indx(strct,1199)].flg_anly= 1

  strct[fnd_indx(strct,1194):fnd_indx(strct,1196)].type    = 'OBJ'
  strct[fnd_indx(strct,1197)].type = 'ARC'
  strct[fnd_indx(strct,1198):fnd_indx(strct,1199)].type    = 'FLT'
  strct[fnd_indx(strct,1193)].type = 'IMG'



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  print, 'Setcrds finished.'


return
end
