;--------------------------------------------------------------------
;+
; NAME:
;     ldss_setup
; 
; PURPOSE:
;    Set up LDSS3 longslit reduction directories, create data structure.
;
; CALLING SEQUENCE:
;    ldss_setup, night
;
; INPUTS:
;    night -- 'obs020205'
;
;    This procedure must be called in a directory which
;    contains /rawdata (all raw data to be processed) and
;    a setcrd file
;
; OUTPUTS:
;
; OPTIONAL OUTPUTS:
;
;
; MODIFICATION HISTORY:
;    most ldss routines are based on prochaska's wfccd package
;    mg 3/05
;- 
;--------------------------------------------------------------------
Pro ldss_setup,night,struct, grism=grism, slitwidth=slitwidth

  if  N_params() LT 2  then begin 
      print,'Syntax - ' + $
        'ldss_setup, night, strct'
      return
  endif 


; SET PARAMETERS NOT IN HEADER -- CAN MODIFY THESE IN SETCRDs
  if NOT keyword_set(grism) then grism = 'VPHblue'
  if NOT keyword_set(slitwidth) then slitwidth = 0.8



; REQUIRES RAW DATA STORE INTO DIRECTORY
  rawdir='rawdata/'
  if(NOT file_test(rawdir)) then begin
    splog, 'no raw data for night'
    return
  endif



; MAKE REDUCTION DIRECTORIES
    a = file_search('Flats/..', count=count)
    if count EQ 0 then file_mkdir, 'Flats'
    a = file_search('Bias/..', count=count)
    if count EQ 0 then file_mkdir, 'Bias'
    a = file_search('Arcs/..', count=count)
    if count EQ 0 then file_mkdir, 'Arcs'
    a = file_search('Final/..', count=count)
    if count EQ 0 then file_mkdir, 'Final'
    a = file_search('Extract/..', count=count)
    if count EQ 0 then file_mkdir, 'Extract'

; NUMBER OF IMAGES  
  img = file_search('rawdata/ccd*c1.fits*') 
  nimg = n_elements(img)

;  nimg = nimg/4.
;  if (nimg 


; MAKE REDUCTION STRUCTURE
  tmp = { ldssstr }
  struct = replicate(tmp,nimg) 


;  LOOP ON INDIVIDUAL IMAGES - ccd***C1 SETS HEADER
  for q=0,nimg-1 do begin
         print, 'Reading ', img[q]
         head = xheadfits(img[q], /silent)
      
       ; DEFAULT ALL IMAGES ARE JUNK :)
         struct[q].flg_anly = 0

       ; SET FRAME NUMBER
         struct[q].ccdframe = img[q]

         x=strsplit(sxpar(head,'FILENAME'),'c',/extract)
         num = strsplit(x[0],'d',/extract)
         struct[q].ccdnum = string(num(0))
         print,struct[q].ccdframe,struct[q].ccdnum

       ; PARSE HEADER
         struct[q].exp  = sxpar(head, 'EXPTIME')
         struct[q].RA   = sxpar(head, 'RA-D')
         struct[q].DEC  = sxpar(head, 'DEC-D')
         struct[q].SRA   = sxpar(head, 'RA')
         struct[q].SDEC  = sxpar(head, 'DEC')
         struct[q].Equinox = sxpar(head, 'EQUINOX')
         struct[q].UT      = sxpar(head, 'UT-TIME')
         struct[q].dateobs = sxpar(head, 'DATE-OBS')
         struct[q].airmass = sxpar(head, 'AIRMASS')
         struct[q].rotang  = sxpar(head, 'ROTANGLE')


       ; SET CCD SPECIFIC -- CHIP 1
         struct[q].chip[0]   = sxpar(head, 'CHIP')
         struct[q].readno[0] = sxpar(head, 'ENOISE')
         struct[q].gain[0]   = sxpar(head, 'EGAIN')

         
       ; CHIPS 2-4
         for j = 1,3 do begin

             print, 'Reading ', img[q]
             head = xheadfits(img[q], /silent)

             struct[q].chip[j]   = sxpar(head, 'CHIP')
             struct[q].readno[j] = sxpar(head, 'ENOISE')
             struct[q].gain[j]   = sxpar(head, 'EGAIN')

        endfor

       ; SET GRATING/GRATING TILT
         struct[q].grism      = grism
         struct[q].slitwidth  = slitwidth
                
      
  endfor

; RUN SETCRDS TO SET INDIVIDUAL EXPOSURES 
  setcrds = 'setcrds_'+night
  print,'running ',setcrds
  call_procedure,setcrds,struct
  struct.type = strcompress(struct.type,/remove_all)  ; had problems with
                                                      ; whitespace 

; Write the structure to FITS
  outfil = 'ldss_'+night+'.fits'
  mwrfits, struct, outfil, /create

; All done
  print, 'ldss_setup: All done!'


end
