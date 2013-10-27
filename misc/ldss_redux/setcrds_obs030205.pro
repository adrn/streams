pro setcrds_obs030205, strct

;  This program sets the structure cards for Mar 2+3, 2005
;  Magellan LDSS3 run


  print, 'Setting the cards for obs030205 data'
  nimg = n_elements(strct)

; BIAS FRAMES
;  strct[fnd_indx(strct,201):fnd_indx(strct,209)].type = 'ZRO'
;  strct[fnd_indx(strct,201):fnd_indx(strct,209)].flg_anly =1

  strct[fnd_indx(strct,359):fnd_indx(strct,369)].type = 'ZRO'
  strct[fnd_indx(strct,359):fnd_indx(strct,369)].flg_anly =1

  strct[fnd_indx(strct,76):fnd_indx(strct,200)].bias_fil = 'Bias/bias_n1'
  strct[fnd_indx(strct,213):fnd_indx(strct,369)].bias_fil = 'Bias/bias_n2'
  

; IMAGE TO DETERMINE GAIN
  strct[fnd_indx(strct,197):fnd_indx(strct,200)].type = 'GFT'
  strct[fnd_indx(strct,197):fnd_indx(strct,200)].flg_anly =1



; OBJECTS -- NIGHT 1
  strct[fnd_indx(strct,76):fnd_indx(strct,80)].objname   = '268302' ; 1
  strct[fnd_indx(strct,81):fnd_indx(strct,88)].objname   = '276603' ; 2
  strct[fnd_indx(strct,89):fnd_indx(strct,97)].objname   = '262647' ; 3
  strct[fnd_indx(strct,98):fnd_indx(strct,105)].objname  = '274726' ; 4
  strct[fnd_indx(strct,106):fnd_indx(strct,113)].objname = '280272' ; 5
  strct[fnd_indx(strct,114):fnd_indx(strct,120)].objname = '67068'  ; 6
  strct[fnd_indx(strct,121):fnd_indx(strct,127)].objname = '158190' ; 7
  strct[fnd_indx(strct,128):fnd_indx(strct,134)].objname = '172859' ; 8
  strct[fnd_indx(strct,135):fnd_indx(strct,142)].objname = '16651'  ; 9
  strct[fnd_indx(strct,143):fnd_indx(strct,146)].objname = '160238' ; 10
  strct[fnd_indx(strct,147):fnd_indx(strct,153)].objname = '73362'  ; 11
  strct[fnd_indx(strct,154):fnd_indx(strct,160)].objname = '10213'  ; 12
  strct[fnd_indx(strct,161):fnd_indx(strct,167)].objname = '370242' ; 13
  strct[fnd_indx(strct,168):fnd_indx(strct,174)].objname = '178719' ; 14
  strct[fnd_indx(strct,175):fnd_indx(strct,182)].objname = '307723' ; 15

; Night 2
  strct[fnd_indx(strct,227):fnd_indx(strct,235)].objname = '161656' ; 16
  strct[fnd_indx(strct,236):fnd_indx(strct,242)].objname = '162130' ; 17
  strct[fnd_indx(strct,243):fnd_indx(strct,251)].objname = '278622' ; 18
  strct[fnd_indx(strct,252):fnd_indx(strct,258)].objname = '266041' ; 19
  strct[fnd_indx(strct,259):fnd_indx(strct,265)].objname = '244342' ; 20
  strct[fnd_indx(strct,266):fnd_indx(strct,272)].objname = '330228' ; 21
  strct[fnd_indx(strct,273):fnd_indx(strct,279)].objname = '163077' ; 22
  strct[fnd_indx(strct,280):fnd_indx(strct,286)].objname = '170118' ; 23
  strct[fnd_indx(strct,287):fnd_indx(strct,294)].objname = '149339' ; 24
  strct[fnd_indx(strct,295):fnd_indx(strct,300)].objname = '148003' ; 25
  strct[fnd_indx(strct,301):fnd_indx(strct,306)].objname = '329114' ; 26
  strct[fnd_indx(strct,307):fnd_indx(strct,313)].objname = '160437' ; 27
  strct[fnd_indx(strct,314):fnd_indx(strct,320)].objname = '6716'   ; 28
  strct[fnd_indx(strct,321):fnd_indx(strct,328)].objname = '157095' ; 29
  strct[fnd_indx(strct,329):fnd_indx(strct,335)].objname = '11160'  ; 30
  strct[fnd_indx(strct,336):fnd_indx(strct,342)].objname = '301586' ; 31
  strct[fnd_indx(strct,343):fnd_indx(strct,349)].objname = '371747' ; 32




;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 82511

 ; objnum=1
 ; strct[fnd_indx(strct,59):fnd_indx(strct,67)].objnum  = objnum
 ; strct[fnd_indx(strct,59):fnd_indx(strct,67)].flg_anly= 1

 ; strct[fnd_indx(strct,63):fnd_indx(strct,64)].type    = 'OBJ'
 ; strct[fnd_indx(strct,65)].type = 'ARC'
 ; strct[fnd_indx(strct,66):fnd_indx(strct,67)].type    = 'FLT'

;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 268302

  objnum=1
  strct[fnd_indx(strct,76):fnd_indx(strct,80)].objnum  = objnum
  strct[fnd_indx(strct,76):fnd_indx(strct,80)].flg_anly= 1

  strct[fnd_indx(strct,76):fnd_indx(strct,77)].type    = 'OBJ'
  strct[fnd_indx(strct,78)].type = 'ARC'
  strct[fnd_indx(strct,79):fnd_indx(strct,80)].type    = 'FLT'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 276603

  objnum=2
  strct[fnd_indx(strct,81):fnd_indx(strct,88)].objnum  = objnum
  strct[fnd_indx(strct,81):fnd_indx(strct,88)].flg_anly= 1

  strct[fnd_indx(strct,84):fnd_indx(strct,85)].type    = 'OBJ'
  strct[fnd_indx(strct,86)].type = 'ARC'
  strct[fnd_indx(strct,87):fnd_indx(strct,88)].type    = 'FLT'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 262647

  objnum=3
  strct[fnd_indx(strct,89):fnd_indx(strct,97)].objnum  = objnum
  strct[fnd_indx(strct,89):fnd_indx(strct,97)].flg_anly= 1

  strct[fnd_indx(strct,93):fnd_indx(strct,94)].type    = 'OBJ'
  strct[fnd_indx(strct,95)].type = 'ARC'
  strct[fnd_indx(strct,96):fnd_indx(strct,97)].type    = 'FLT'
; requires hack in crflag tmp[3375:3405,445:460] = 0
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 274726

  objnum=4
  strct[fnd_indx(strct,98):fnd_indx(strct,105)].objnum  = objnum
  strct[fnd_indx(strct,98):fnd_indx(strct,105)].flg_anly= 1

  strct[fnd_indx(strct,100):fnd_indx(strct,102)].type    = 'OBJ'
  strct[fnd_indx(strct,103)].type = 'ARC'
  strct[fnd_indx(strct,104):fnd_indx(strct,105)].type    = 'FLT'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 280272

  objnum=5
  strct[fnd_indx(strct,106):fnd_indx(strct,113)].objnum  = objnum
  strct[fnd_indx(strct,106):fnd_indx(strct,113)].flg_anly= 1

  strct[fnd_indx(strct,109):fnd_indx(strct,110)].type    = 'OBJ'
  strct[fnd_indx(strct,111)].type = 'ARC'
  strct[fnd_indx(strct,112):fnd_indx(strct,113)].type    = 'FLT'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 67068

  objnum=6
  strct[fnd_indx(strct,114):fnd_indx(strct,120)].objnum  = objnum
  strct[fnd_indx(strct,114):fnd_indx(strct,120)].flg_anly= 1

  strct[fnd_indx(strct,116):fnd_indx(strct,117)].type    = 'OBJ'
  strct[fnd_indx(strct,118)].type = 'ARC'
  strct[fnd_indx(strct,119):fnd_indx(strct,120)].type    = 'FLT'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 158190

  objnum=7
  strct[fnd_indx(strct,121):fnd_indx(strct,127)].objnum  = objnum
  strct[fnd_indx(strct,121):fnd_indx(strct,127)].flg_anly= 1

  strct[fnd_indx(strct,123):fnd_indx(strct,124)].type    = 'OBJ'
  strct[fnd_indx(strct,125)].type = 'ARC'
  strct[fnd_indx(strct,126):fnd_indx(strct,127)].type    = 'FLT'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 172859

  objnum=8
  strct[fnd_indx(strct,128):fnd_indx(strct,134)].objnum  = objnum
  strct[fnd_indx(strct,128):fnd_indx(strct,134)].flg_anly= 1

  strct[fnd_indx(strct,130):fnd_indx(strct,131)].type    = 'OBJ'
  strct[fnd_indx(strct,132)].type = 'ARC'
  strct[fnd_indx(strct,133):fnd_indx(strct,134)].type    = 'FLT'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 16651

  objnum=9
  strct[fnd_indx(strct,135):fnd_indx(strct,142)].objnum  = objnum
  strct[fnd_indx(strct,135):fnd_indx(strct,142)].flg_anly= 1

  strct[fnd_indx(strct,138):fnd_indx(strct,139)].type    = 'OBJ'
  strct[fnd_indx(strct,140)].type = 'ARC'
  strct[fnd_indx(strct,141):fnd_indx(strct,142)].type    = 'FLT'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 160238

  objnum=10
  strct[fnd_indx(strct,143):fnd_indx(strct,146)].objnum  = objnum
  strct[fnd_indx(strct,143):fnd_indx(strct,146)].flg_anly= 1

  strct[fnd_indx(strct,145)].type    = 'OBJ'
  strct[fnd_indx(strct,146)].type    = 'FLT' 

  strct[fnd_indx(strct,370)].flg_anly= 1
  strct[fnd_indx(strct,370)].objnum  = objnum
  strct[fnd_indx(strct,370)].type = 'ARC'    ;  made-up arc for this object
                                             ;  really ccd0140.

                            
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 73362

  objnum=11
  strct[fnd_indx(strct,147):fnd_indx(strct,153)].objnum  = objnum
  strct[fnd_indx(strct,147):fnd_indx(strct,153)].flg_anly= 1

  strct[fnd_indx(strct,149):fnd_indx(strct,150)].type    = 'OBJ'
  strct[fnd_indx(strct,151)].type = 'ARC'
  strct[fnd_indx(strct,152):fnd_indx(strct,153)].type    = 'FLT'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 10213

  objnum=12
  strct[fnd_indx(strct,154):fnd_indx(strct,160)].objnum  = objnum
  strct[fnd_indx(strct,154):fnd_indx(strct,160)].flg_anly= 1

  strct[fnd_indx(strct,156):fnd_indx(strct,157)].type    = 'OBJ'
  strct[fnd_indx(strct,158)].type = 'ARC'
  strct[fnd_indx(strct,159):fnd_indx(strct,160)].type    = 'FLT'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 370242

  objnum=13
  strct[fnd_indx(strct,161):fnd_indx(strct,167)].objnum  = objnum
  strct[fnd_indx(strct,161):fnd_indx(strct,167)].flg_anly= 1

  strct[fnd_indx(strct,163):fnd_indx(strct,164)].type    = 'OBJ'
  strct[fnd_indx(strct,165)].type = 'ARC'
  strct[fnd_indx(strct,166):fnd_indx(strct,167)].type    = 'FLT'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 178719

  objnum=14
  strct[fnd_indx(strct,168):fnd_indx(strct,174)].objnum  = objnum
  strct[fnd_indx(strct,168):fnd_indx(strct,174)].flg_anly= 1

  strct[fnd_indx(strct,170):fnd_indx(strct,171)].type    = 'OBJ'
  strct[fnd_indx(strct,172)].type = 'ARC'
  strct[fnd_indx(strct,173):fnd_indx(strct,174)].type    = 'FLT'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 307723

  objnum=15
  strct[fnd_indx(strct,176):fnd_indx(strct,182)].objnum  = objnum
  strct[fnd_indx(strct,176):fnd_indx(strct,182)].flg_anly= 1

  strct[fnd_indx(strct,178):fnd_indx(strct,179)].type    = 'OBJ'
  strct[fnd_indx(strct,180)].type = 'ARC'
  strct[fnd_indx(strct,181):fnd_indx(strct,182)].type    = 'FLT'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 161656

  objnum=16
  strct[fnd_indx(strct,227):fnd_indx(strct,235)].objnum  = objnum
  strct[fnd_indx(strct,227):fnd_indx(strct,235)].flg_anly= 1

  strct[fnd_indx(strct,230)].type    = 'OBJ'    ; vingetter issue
  strct[fnd_indx(strct,235)].type    = 'OBJ'    ; vingetter issue
  strct[fnd_indx(strct,232)].type = 'ARC'
  strct[fnd_indx(strct,233):fnd_indx(strct,234)].type    = 'FLT'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 162130

  objnum=17
  strct[fnd_indx(strct,236):fnd_indx(strct,242)].objnum  = objnum
  strct[fnd_indx(strct,236):fnd_indx(strct,242)].flg_anly= 1

  strct[fnd_indx(strct,238):fnd_indx(strct,239)].type    = 'OBJ'
  strct[fnd_indx(strct,240)].type = 'ARC'
  strct[fnd_indx(strct,241):fnd_indx(strct,242)].type    = 'FLT'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 278622

  objnum=18
  strct[fnd_indx(strct,243):fnd_indx(strct,251)].objnum  = objnum
  strct[fnd_indx(strct,243):fnd_indx(strct,251)].flg_anly= 1

  strct[fnd_indx(strct,247):fnd_indx(strct,248)].type    = 'OBJ'
  strct[fnd_indx(strct,249)].type = 'ARC'
  strct[fnd_indx(strct,250):fnd_indx(strct,251)].type    = 'FLT'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 266041 

  objnum=19
  strct[fnd_indx(strct,252):fnd_indx(strct,258)].objnum  = objnum
  strct[fnd_indx(strct,252):fnd_indx(strct,258)].flg_anly= 1

  strct[fnd_indx(strct,254):fnd_indx(strct,255)].type    = 'OBJ'
  strct[fnd_indx(strct,256)].type = 'ARC'
  strct[fnd_indx(strct,257):fnd_indx(strct,258)].type    = 'FLT'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 244342

  objnum=20
  strct[fnd_indx(strct,259):fnd_indx(strct,265)].objnum  = objnum
  strct[fnd_indx(strct,259):fnd_indx(strct,265)].flg_anly= 1

  strct[fnd_indx(strct,261):fnd_indx(strct,262)].type    = 'OBJ'
  strct[fnd_indx(strct,263)].type = 'ARC'
  strct[fnd_indx(strct,264):fnd_indx(strct,265)].type    = 'FLT'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 330228

  objnum=21
  strct[fnd_indx(strct,266):fnd_indx(strct,272)].objnum  = objnum
  strct[fnd_indx(strct,266):fnd_indx(strct,272)].flg_anly= 1

  strct[fnd_indx(strct,268):fnd_indx(strct,269)].type    = 'OBJ'
  strct[fnd_indx(strct,270)].type = 'ARC'    ; had to fudge this arc w/263
  strct[fnd_indx(strct,271):fnd_indx(strct,272)].type    = 'FLT'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 163077

  objnum=22
  strct[fnd_indx(strct,273):fnd_indx(strct,279)].objnum  = objnum
  strct[fnd_indx(strct,273):fnd_indx(strct,279)].flg_anly= 1

  strct[fnd_indx(strct,275):fnd_indx(strct,276)].type    = 'OBJ'
  strct[fnd_indx(strct,277)].type = 'ARC'
  strct[fnd_indx(strct,278):fnd_indx(strct,279)].type    = 'FLT'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 170118

  objnum=23
  strct[fnd_indx(strct,280):fnd_indx(strct,286)].objnum  = objnum
  strct[fnd_indx(strct,280):fnd_indx(strct,286)].flg_anly= 1

  strct[fnd_indx(strct,282):fnd_indx(strct,283)].type    = 'OBJ'
  strct[fnd_indx(strct,284)].type = 'ARC'
  strct[fnd_indx(strct,285):fnd_indx(strct,286)].type    = 'FLT'

;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 149399

  objnum=24
  strct[fnd_indx(strct,287):fnd_indx(strct,294)].objnum  = objnum
  strct[fnd_indx(strct,287):fnd_indx(strct,294)].flg_anly= 1

  strct[fnd_indx(strct,290):fnd_indx(strct,291)].type    = 'OBJ'
  strct[fnd_indx(strct,292)].type = 'ARC'
  strct[fnd_indx(strct,293):fnd_indx(strct,294)].type    = 'FLT'

;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 148003

  objnum=25
  strct[fnd_indx(strct,295):fnd_indx(strct,300)].objnum  = objnum
  strct[fnd_indx(strct,295):fnd_indx(strct,300)].flg_anly= 1

  strct[fnd_indx(strct,296):fnd_indx(strct,297)].type    = 'OBJ'
  strct[fnd_indx(strct,298)].type = 'ARC'
  strct[fnd_indx(strct,299):fnd_indx(strct,300)].type    = 'FLT'

;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 329114

  objnum=26
  strct[fnd_indx(strct,301):fnd_indx(strct,306)].objnum  = objnum
  strct[fnd_indx(strct,301):fnd_indx(strct,306)].flg_anly= 1

  strct[fnd_indx(strct,302):fnd_indx(strct,303)].type    = 'OBJ'
  strct[fnd_indx(strct,304)].type = 'ARC'
  strct[fnd_indx(strct,305):fnd_indx(strct,306)].type    = 'FLT'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 160437

  objnum=27
  strct[fnd_indx(strct,307):fnd_indx(strct,313)].objnum  = objnum
  strct[fnd_indx(strct,307):fnd_indx(strct,313)].flg_anly= 1

  strct[fnd_indx(strct,309):fnd_indx(strct,310)].type    = 'OBJ'
  strct[fnd_indx(strct,311)].type = 'ARC'
  strct[fnd_indx(strct,312):fnd_indx(strct,313)].type    = 'FLT'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 6716

  objnum=28
  strct[fnd_indx(strct,314):fnd_indx(strct,320)].objnum  = objnum
  strct[fnd_indx(strct,314):fnd_indx(strct,320)].flg_anly= 1

  strct[fnd_indx(strct,316):fnd_indx(strct,317)].type    = 'OBJ'
  strct[fnd_indx(strct,318)].type = 'ARC'
  strct[fnd_indx(strct,319):fnd_indx(strct,320)].type    = 'FLT'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 157095

  objnum=29
  strct[fnd_indx(strct,321):fnd_indx(strct,328)].objnum  = objnum
  strct[fnd_indx(strct,321):fnd_indx(strct,328)].flg_anly= 1

  strct[fnd_indx(strct,323):fnd_indx(strct,324)].type    = 'OBJ'
  strct[fnd_indx(strct,325)].type = 'ARC'
  strct[fnd_indx(strct,326):fnd_indx(strct,327)].type    = 'FLT'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 11160

  objnum=30
  strct[fnd_indx(strct,329):fnd_indx(strct,335)].objnum  = objnum
  strct[fnd_indx(strct,329):fnd_indx(strct,335)].flg_anly= 1

  strct[fnd_indx(strct,331):fnd_indx(strct,332)].type    = 'OBJ'
  strct[fnd_indx(strct,333)].type = 'ARC'
  strct[fnd_indx(strct,334):fnd_indx(strct,335)].type    = 'FLT'
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 301586

  objnum=31
  strct[fnd_indx(strct,336):fnd_indx(strct,342)].objnum  = objnum
  strct[fnd_indx(strct,336):fnd_indx(strct,342)].flg_anly= 1

  strct[fnd_indx(strct,338):fnd_indx(strct,339)].type    = 'OBJ'
  strct[fnd_indx(strct,340)].type = 'ARC'
  strct[fnd_indx(strct,341):fnd_indx(strct,342)].type    = 'FLT'

; need hack in crflag tmp[3400:3425,460:475] = 0

;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; 371747

  objnum=32
  strct[fnd_indx(strct,343):fnd_indx(strct,349)].objnum  = objnum
  strct[fnd_indx(strct,343):fnd_indx(strct,349)].flg_anly= 1

  strct[fnd_indx(strct,345):fnd_indx(strct,346)].type    = 'OBJ'
  strct[fnd_indx(strct,347)].type = 'ARC'
  strct[fnd_indx(strct,348):fnd_indx(strct,349)].type    = 'FLT'

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;



  print, 'Setcrds finished.'


return
end
