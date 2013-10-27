pro setcrds_obs080305, strct

;  This program sets the structure cards for August 3, 2005
;  Magellan LDSS3 run


  print, 'Setting the cards for obs080305 data'
  nimg = n_elements(strct)

; BIAS FRAMES
  strct[fnd_indx(strct,47):fnd_indx(strct,51)].type = 'ZRO'
  strct[fnd_indx(strct,47):fnd_indx(strct,51)].flg_anly =1
  strct[fnd_indx(strct,1):fnd_indx(strct,51)].bias_fil = 'Bias/bias'
  

; IMAGE TO DETERMINE GAIN
  strct[fnd_indx(strct,61):fnd_indx(strct,65)].type = 'GFT'
  strct[fnd_indx(strct,61):fnd_indx(strct,65)].flg_anly =1



; OBJECTS -- NIGHT 2
  strct[fnd_indx(strct,4):fnd_indx(strct,10)].objname   = '677002'   ; 33
  strct[fnd_indx(strct,11):fnd_indx(strct,18)].objname   = '677307'  ; 34
  strct[fnd_indx(strct,19):fnd_indx(strct,26)].objname   = '222989'  ; 35
  strct[fnd_indx(strct,27):fnd_indx(strct,33)].objname  = '461714'   ; 36
  strct[fnd_indx(strct,34):fnd_indx(strct,40)].objname = '191180'    ; 37




;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 677002

  objnum=33
  strct[fnd_indx(strct,4):fnd_indx(strct,10)].objnum  = objnum
  strct[fnd_indx(strct,6):fnd_indx(strct,10)].flg_anly= 1

  strct[fnd_indx(strct,6):fnd_indx(strct,7)].type    = 'OBJ'
  strct[fnd_indx(strct,8):fnd_indx(strct,9)].type    = 'FLT'
  strct[fnd_indx(strct,10)].type = 'ARC'

;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 677307

  objnum=34
  strct[fnd_indx(strct,11):fnd_indx(strct,18)].objnum  = objnum
  strct[fnd_indx(strct,14):fnd_indx(strct,18)].flg_anly= 1

  strct[fnd_indx(strct,14):fnd_indx(strct,15)].type    = 'OBJ'
  strct[fnd_indx(strct,16):fnd_indx(strct,17)].type    = 'FLT'
  strct[fnd_indx(strct,18)].type = 'ARC'

;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 222989

  objnum=35
  strct[fnd_indx(strct,19):fnd_indx(strct,26)].objnum  = objnum
  strct[fnd_indx(strct,22):fnd_indx(strct,26)].flg_anly= 1

  strct[fnd_indx(strct,22):fnd_indx(strct,23)].type    = 'OBJ'
  strct[fnd_indx(strct,24):fnd_indx(strct,25)].type    = 'FLT'
  strct[fnd_indx(strct,26)].type = 'ARC'

;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 461714

  objnum=36
  strct[fnd_indx(strct,27):fnd_indx(strct,33)].objnum  = objnum
  strct[fnd_indx(strct,29):fnd_indx(strct,33)].flg_anly= 1

  strct[fnd_indx(strct,29):fnd_indx(strct,30)].type    = 'OBJ'
  strct[fnd_indx(strct,31):fnd_indx(strct,32)].type    = 'FLT'
  strct[fnd_indx(strct,33)].type = 'ARC'

;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 191180

  objnum=37
  strct[fnd_indx(strct,34):fnd_indx(strct,40)].objnum  = objnum
  strct[fnd_indx(strct,36):fnd_indx(strct,40)].flg_anly= 1

  strct[fnd_indx(strct,36):fnd_indx(strct,37)].type    = 'OBJ'
  strct[fnd_indx(strct,38):fnd_indx(strct,39)].type    = 'FLT'
  strct[fnd_indx(strct,40)].type = 'ARC'





  print, 'Setcrds finished.'


return
end
