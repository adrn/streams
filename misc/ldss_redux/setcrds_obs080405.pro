pro setcrds_obs080405, strct

;  This program sets the structure cards for August 4, 2005
;  Magellan LDSS3 run


  print, 'Setting the cards for obs080405 data'
  nimg = n_elements(strct)

; BIAS FRAMES
  strct[fnd_indx(strct,52):fnd_indx(strct,56)].type = 'ZRO'
  strct[fnd_indx(strct,56):fnd_indx(strct,56)].flg_anly =1
  strct[fnd_indx(strct,56):fnd_indx(strct,192)].bias_fil = 'Bias/bias'

  strct[fnd_indx(strct,182):fnd_indx(strct,192)].objname = 'LTT1020'   

; IMAGE TO DETERMINE GAIN
  strct[fnd_indx(strct,61):fnd_indx(strct,65)].type = 'GFT'
  strct[fnd_indx(strct,61):fnd_indx(strct,65)].flg_anly =1



; OBJECTS -- NIGHT 2
  strct[fnd_indx(strct,66):fnd_indx(strct,73)].objname   = '167872' ; 38
  strct[fnd_indx(strct,74):fnd_indx(strct,81)].objname   = '154344' ; 39
  strct[fnd_indx(strct,82):fnd_indx(strct,88)].objname   = '42360'  ; 40
  strct[fnd_indx(strct,89):fnd_indx(strct,94)].objname   = '11663'  ; 41
  strct[fnd_indx(strct,95):fnd_indx(strct,100)].objname  = '1076'   ; 42
  strct[fnd_indx(strct,101):fnd_indx(strct,107)].objname = '399906' ; 43
  strct[fnd_indx(strct,108):fnd_indx(strct,114)].objname = '217927' ; 44  
  strct[fnd_indx(strct,115):fnd_indx(strct,122)].objname = '189157' ; 45
  strct[fnd_indx(strct,125):fnd_indx(strct,130)].objname = '181782' ; 46
  strct[fnd_indx(strct,131):fnd_indx(strct,136)].objname = '181810' ; 47  
  strct[fnd_indx(strct,137):fnd_indx(strct,142)].objname = '410317' ; 48
  strct[fnd_indx(strct,143):fnd_indx(strct,149)].objname = '633141' ; 49  
  strct[fnd_indx(strct,150):fnd_indx(strct,155)].objname = '190416' ; 50
  strct[fnd_indx(strct,156):fnd_indx(strct,161)].objname = '192963' ; 51  high-z companion
  strct[fnd_indx(strct,162):fnd_indx(strct,167)].objname = '446610' ; 52
  strct[fnd_indx(strct,169):fnd_indx(strct,174)].objname = '201346' ; 53
  strct[fnd_indx(strct,175):fnd_indx(strct,180)].objname = '198917' ; 54 *




;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 167872

  objnum=38
  strct[fnd_indx(strct,66):fnd_indx(strct,73)].objnum  = objnum
  strct[fnd_indx(strct,68):fnd_indx(strct,73)].flg_anly= 1

  strct[fnd_indx(strct,71):fnd_indx(strct,72)].type    = 'OBJ'
  strct[fnd_indx(strct,68):fnd_indx(strct,69)].type    = 'FLT'
  strct[fnd_indx(strct,73)].type = 'ARC'

;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 154344

  objnum=39
  strct[fnd_indx(strct,74):fnd_indx(strct,81)].objnum  = objnum
  strct[fnd_indx(strct,77):fnd_indx(strct,81)].flg_anly= 1

  strct[fnd_indx(strct,77):fnd_indx(strct,78)].type    = 'OBJ'
  strct[fnd_indx(strct,79):fnd_indx(strct,80)].type    = 'FLT'
  strct[fnd_indx(strct,81)].type = 'ARC'

;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 42360

  objnum=40
  strct[fnd_indx(strct,82):fnd_indx(strct,88)].objnum  = objnum
  strct[fnd_indx(strct,84):fnd_indx(strct,88)].flg_anly= 1

  strct[fnd_indx(strct,84):fnd_indx(strct,85)].type    = 'OBJ'
  strct[fnd_indx(strct,86):fnd_indx(strct,87)].type    = 'FLT'
  strct[fnd_indx(strct,88)].type = 'ARC'

;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 11663

  objnum=41
  strct[fnd_indx(strct,89):fnd_indx(strct,94)].objnum  = objnum
  strct[fnd_indx(strct,90):fnd_indx(strct,94)].flg_anly= 1

  strct[fnd_indx(strct,90):fnd_indx(strct,91)].type    = 'OBJ'
  strct[fnd_indx(strct,92):fnd_indx(strct,93)].type    = 'FLT'
  strct[fnd_indx(strct,94)].type = 'ARC'

;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 1076

  objnum=42
  strct[fnd_indx(strct,95):fnd_indx(strct,100)].objnum  = objnum
  strct[fnd_indx(strct,96):fnd_indx(strct,100)].flg_anly= 1

  strct[fnd_indx(strct,96):fnd_indx(strct,97)].type    = 'OBJ'
  strct[fnd_indx(strct,98):fnd_indx(strct,99)].type    = 'FLT'
  strct[fnd_indx(strct,100)].type = 'ARC'

;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 399906

  objnum=43
  strct[fnd_indx(strct,101):fnd_indx(strct,107)].objnum  = objnum
  strct[fnd_indx(strct,103):fnd_indx(strct,107)].flg_anly= 1

  strct[fnd_indx(strct,103):fnd_indx(strct,104)].type    = 'OBJ'
  strct[fnd_indx(strct,105):fnd_indx(strct,106)].type    = 'FLT'
  strct[fnd_indx(strct,107)].type = 'ARC'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 217927

  objnum=44
  strct[fnd_indx(strct,108):fnd_indx(strct,114)].objnum  = objnum
  strct[fnd_indx(strct,110):fnd_indx(strct,114)].flg_anly= 1

  strct[fnd_indx(strct,110):fnd_indx(strct,111)].type    = 'OBJ'
  strct[fnd_indx(strct,112):fnd_indx(strct,113)].type    = 'FLT'
  strct[fnd_indx(strct,114)].type = 'ARC'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 189157

  objnum=45
  strct[fnd_indx(strct,115):fnd_indx(strct,122)].objnum  = objnum
  strct[fnd_indx(strct,118):fnd_indx(strct,122)].flg_anly= 1

  strct[fnd_indx(strct,118):fnd_indx(strct,119)].type    = 'OBJ'
  strct[fnd_indx(strct,120):fnd_indx(strct,121)].type    = 'FLT'
  strct[fnd_indx(strct,122)].type = 'ARC'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 181782

  objnum=46
  strct[fnd_indx(strct,125):fnd_indx(strct,130)].objnum  = objnum
  strct[fnd_indx(strct,126):fnd_indx(strct,130)].flg_anly= 1

  strct[fnd_indx(strct,126):fnd_indx(strct,127)].type    = 'OBJ'
  strct[fnd_indx(strct,128):fnd_indx(strct,129)].type    = 'FLT'
  strct[fnd_indx(strct,130)].type = 'ARC'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 181810

  objnum=47
  strct[fnd_indx(strct,131):fnd_indx(strct,136)].objnum  = objnum
  strct[fnd_indx(strct,132):fnd_indx(strct,136)].flg_anly= 1

  strct[fnd_indx(strct,132):fnd_indx(strct,133)].type    = 'OBJ'
  strct[fnd_indx(strct,134):fnd_indx(strct,135)].type    = 'FLT'
  strct[fnd_indx(strct,136)].type = 'ARC'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 410317

  objnum=48
  strct[fnd_indx(strct,137):fnd_indx(strct,142)].objnum  = objnum
  strct[fnd_indx(strct,138):fnd_indx(strct,142)].flg_anly= 1

  strct[fnd_indx(strct,138):fnd_indx(strct,139)].type    = 'OBJ'
  strct[fnd_indx(strct,140):fnd_indx(strct,141)].type    = 'FLT'
  strct[fnd_indx(strct,142)].type = 'ARC'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 633141

  objnum=49
  strct[fnd_indx(strct,143):fnd_indx(strct,149)].objnum  = objnum
  strct[fnd_indx(strct,145):fnd_indx(strct,149)].flg_anly= 1

  strct[fnd_indx(strct,145):fnd_indx(strct,146)].type    = 'OBJ'
  strct[fnd_indx(strct,147):fnd_indx(strct,148)].type    = 'FLT'
  strct[fnd_indx(strct,149)].type = 'ARC'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 190416

  objnum=50
  strct[fnd_indx(strct,150):fnd_indx(strct,155)].objnum  = objnum
  strct[fnd_indx(strct,151):fnd_indx(strct,155)].flg_anly= 1

  strct[fnd_indx(strct,151):fnd_indx(strct,152)].type    = 'OBJ'
  strct[fnd_indx(strct,153):fnd_indx(strct,154)].type    = 'FLT'
  strct[fnd_indx(strct,155)].type = 'ARC'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 191963

  objnum=51
  strct[fnd_indx(strct,156):fnd_indx(strct,161)].objnum  = objnum
  strct[fnd_indx(strct,157):fnd_indx(strct,161)].flg_anly= 1

  strct[fnd_indx(strct,157):fnd_indx(strct,158)].type    = 'OBJ'
  strct[fnd_indx(strct,159):fnd_indx(strct,160)].type    = 'FLT'
  strct[fnd_indx(strct,161)].type = 'ARC'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 446610

  objnum=52
  strct[fnd_indx(strct,162):fnd_indx(strct,167)].objnum  = objnum
  strct[fnd_indx(strct,162):fnd_indx(strct,167)].flg_anly= 1

  strct[fnd_indx(strct,163):fnd_indx(strct,164)].type    = 'OBJ'
  strct[fnd_indx(strct,165):fnd_indx(strct,166)].type    = 'FLT'
  strct[fnd_indx(strct,167)].type = 'ARC'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 201346

  objnum=53
  strct[fnd_indx(strct,169):fnd_indx(strct,174)].objnum  = objnum
  strct[fnd_indx(strct,170):fnd_indx(strct,174)].flg_anly= 1

  strct[fnd_indx(strct,170):fnd_indx(strct,171)].type    = 'OBJ'
  strct[fnd_indx(strct,172):fnd_indx(strct,173)].type    = 'FLT'
  strct[fnd_indx(strct,174)].type = 'ARC'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 198917

  objnum=54
  strct[fnd_indx(strct,175):fnd_indx(strct,180)].objnum  = objnum
  strct[fnd_indx(strct,176):fnd_indx(strct,180)].flg_anly= 1

  strct[fnd_indx(strct,176):fnd_indx(strct,177)].type    = 'OBJ'
  strct[fnd_indx(strct,178):fnd_indx(strct,179)].type    = 'FLT'
  strct[fnd_indx(strct,180)].type = 'ARC'


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  print, 'Setcrds finished.'


return
end
