pro setcrds_obs080505, strct

;  This program sets the structure cards for August 5, 2005
;  Magellan LDSS3 run


  print, 'Setting the cards for obs080505 data'
  nimg = n_elements(strct)

; BIAS FRAMES
  strct[fnd_indx(strct,193):fnd_indx(strct,198)].type = 'ZRO'
  strct[fnd_indx(strct,193):fnd_indx(strct,198)].flg_anly =1
  strct[fnd_indx(strct,193):fnd_indx(strct,313)].bias_fil = 'Bias/bias'





; OBJECTS -- NIGHT 2
  strct[fnd_indx(strct,206):fnd_indx(strct,211)].objname   = '168031'  ; 55 
  strct[fnd_indx(strct,212):fnd_indx(strct,219)].objname   = '11400'   ; 56
  strct[fnd_indx(strct,220):fnd_indx(strct,227)].objname   = '179291'  ; 57
  strct[fnd_indx(strct,228):fnd_indx(strct,234)].objname   = '11663'   ; 58
  strct[fnd_indx(strct,235):fnd_indx(strct,241)].objname   = '188703'  ; 59 
  strct[fnd_indx(strct,242):fnd_indx(strct,247)].objname   = '205988'  ; 60
  strct[fnd_indx(strct,248):fnd_indx(strct,253)].objname   = '633306'  ; 61
  strct[fnd_indx(strct,258):fnd_indx(strct,266)].objname   = '421787'  ; 62
  strct[fnd_indx(strct,267):fnd_indx(strct,273)].objname   = '428358'  ; 63
  strct[fnd_indx(strct,274):fnd_indx(strct,281)].objname   = '195693'  ; 64
  strct[fnd_indx(strct,282):fnd_indx(strct,288)].objname   = '198542'  ; 65
  strct[fnd_indx(strct,289):fnd_indx(strct,294)].objname   = '677387'  ; 66
  strct[fnd_indx(strct,295):fnd_indx(strct,300)].objname   = '231588'  ; 67
  strct[fnd_indx(strct,301):fnd_indx(strct,306)].objname   = '198935'  ; 68
  strct[fnd_indx(strct,307):fnd_indx(strct,313)].objname   = '643582'  ; 69


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 168031

  objnum=55
  strct[fnd_indx(strct,206):fnd_indx(strct,211)].objnum  = objnum
  strct[fnd_indx(strct,206):fnd_indx(strct,211)].flg_anly= 1

  strct[fnd_indx(strct,209):fnd_indx(strct,210)].type    = 'OBJ'
  strct[fnd_indx(strct,206):fnd_indx(strct,207)].type    = 'FLT'
  strct[fnd_indx(strct,211)].type = 'ARC'

;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 11400

  objnum=56
  strct[fnd_indx(strct,214):fnd_indx(strct,219)].objnum  = objnum
  strct[fnd_indx(strct,215):fnd_indx(strct,219)].flg_anly= 1

  strct[fnd_indx(strct,215):fnd_indx(strct,216)].type    = 'OBJ'
  strct[fnd_indx(strct,217):fnd_indx(strct,218)].type    = 'FLT'
  strct[fnd_indx(strct,219)].type = 'ARC'

;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 179291

  objnum=57
  strct[fnd_indx(strct,220):fnd_indx(strct,227)].objnum  = objnum
  strct[fnd_indx(strct,223):fnd_indx(strct,227)].flg_anly= 1

  strct[fnd_indx(strct,223):fnd_indx(strct,224)].type    = 'OBJ'
  strct[fnd_indx(strct,226):fnd_indx(strct,227)].type    = 'FLT'
  strct[fnd_indx(strct,225)].type = 'ARC'

;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 11663

  objnum=58
  strct[fnd_indx(strct,228):fnd_indx(strct,234)].objnum  = objnum
  strct[fnd_indx(strct,230):fnd_indx(strct,234)].flg_anly= 1

  strct[fnd_indx(strct,230):fnd_indx(strct,231)].type    = 'OBJ'
  strct[fnd_indx(strct,233):fnd_indx(strct,234)].type    = 'FLT'
  strct[fnd_indx(strct,232)].type = 'ARC'

;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 188703

  objnum=59
  strct[fnd_indx(strct,235):fnd_indx(strct,241)].objnum  = objnum
  strct[fnd_indx(strct,237):fnd_indx(strct,241)].flg_anly= 1

  strct[fnd_indx(strct,237):fnd_indx(strct,238)].type    = 'OBJ'
  strct[fnd_indx(strct,239):fnd_indx(strct,240)].type    = 'FLT'
  strct[fnd_indx(strct,241)].type = 'ARC'

;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 205988

  objnum=60
  strct[fnd_indx(strct,242):fnd_indx(strct,247)].objnum  = objnum
  strct[fnd_indx(strct,243):fnd_indx(strct,247)].flg_anly= 1

  strct[fnd_indx(strct,243):fnd_indx(strct,244)].type    = 'OBJ'
  strct[fnd_indx(strct,245):fnd_indx(strct,246)].type    = 'FLT'
  strct[fnd_indx(strct,247)].type = 'ARC'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 633306

  objnum=61
  strct[fnd_indx(strct,248):fnd_indx(strct,253)].objnum  = objnum
  strct[fnd_indx(strct,249):fnd_indx(strct,253)].flg_anly= 1

  strct[fnd_indx(strct,249):fnd_indx(strct,250)].type    = 'OBJ'
  strct[fnd_indx(strct,251):fnd_indx(strct,252)].type    = 'FLT'
  strct[fnd_indx(strct,253)].type = 'ARC'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 421787

  objnum=62
  strct[fnd_indx(strct,258):fnd_indx(strct,266)].objnum  = objnum
  strct[fnd_indx(strct,262):fnd_indx(strct,266)].flg_anly= 1

  strct[fnd_indx(strct,262):fnd_indx(strct,263)].type    = 'OBJ'
  strct[fnd_indx(strct,264):fnd_indx(strct,265)].type    = 'FLT'
  strct[fnd_indx(strct,266)].type = 'ARC'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 428358

  objnum=63
  strct[fnd_indx(strct,267):fnd_indx(strct,273)].objnum  = objnum
  strct[fnd_indx(strct,269):fnd_indx(strct,273)].flg_anly= 1

  strct[fnd_indx(strct,269):fnd_indx(strct,270)].type    = 'OBJ'
  strct[fnd_indx(strct,271):fnd_indx(strct,272)].type    = 'FLT'
  strct[fnd_indx(strct,273)].type = 'ARC'

;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 195693

  objnum=64
  strct[fnd_indx(strct,274):fnd_indx(strct,281)].objnum  = objnum
  strct[fnd_indx(strct,276):fnd_indx(strct,281)].flg_anly= 1

  strct[fnd_indx(strct,276):fnd_indx(strct,278)].type    = 'OBJ'
  strct[fnd_indx(strct,279):fnd_indx(strct,280)].type    = 'FLT'
  strct[fnd_indx(strct,281)].type = 'ARC'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 198542

  objnum=65
  strct[fnd_indx(strct,282):fnd_indx(strct,288)].objnum  = objnum
  strct[fnd_indx(strct,283):fnd_indx(strct,287)].flg_anly= 1

  strct[fnd_indx(strct,283):fnd_indx(strct,284)].type    = 'OBJ'
  strct[fnd_indx(strct,285):fnd_indx(strct,286)].type    = 'FLT'
  strct[fnd_indx(strct,287)].type = 'ARC'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 677387

  objnum=66
  strct[fnd_indx(strct,289):fnd_indx(strct,294)].objnum  = objnum
  strct[fnd_indx(strct,290):fnd_indx(strct,294)].flg_anly= 1

  strct[fnd_indx(strct,290):fnd_indx(strct,291)].type    = 'OBJ'
  strct[fnd_indx(strct,292):fnd_indx(strct,293)].type    = 'FLT'
  strct[fnd_indx(strct,294)].type = 'ARC'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 231588

  objnum=67
  strct[fnd_indx(strct,295):fnd_indx(strct,300)].objnum  = objnum
  strct[fnd_indx(strct,296):fnd_indx(strct,300)].flg_anly= 1

  strct[fnd_indx(strct,296):fnd_indx(strct,297)].type    = 'OBJ'
  strct[fnd_indx(strct,298):fnd_indx(strct,299)].type    = 'FLT'
  strct[fnd_indx(strct,300)].type = 'ARC'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 198935

  objnum=68
  strct[fnd_indx(strct,301):fnd_indx(strct,306)].objnum  = objnum
  strct[fnd_indx(strct,302):fnd_indx(strct,306)].flg_anly= 1

  strct[fnd_indx(strct,302):fnd_indx(strct,303)].type    = 'OBJ'
  strct[fnd_indx(strct,304):fnd_indx(strct,305)].type    = 'FLT'
  strct[fnd_indx(strct,306)].type = 'ARC'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 643582

  objnum=69
  strct[fnd_indx(strct,307):fnd_indx(strct,313)].objnum  = objnum
  strct[fnd_indx(strct,308):fnd_indx(strct,313)].flg_anly= 1

  strct[fnd_indx(strct,308):fnd_indx(strct,309)].type    = 'OBJ'
  strct[fnd_indx(strct,310):fnd_indx(strct,311)].type    = 'FLT'
  strct[fnd_indx(strct,313)].type = 'ARC'
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  print, 'Setcrds finished.'


return
end
