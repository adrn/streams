;--------------------------------------------------------------------
;+
; NAME:
;      ldss_stitch2
; 
; PURPOSE:
;     Bias subtract, trim and stitch together an LDSS3 image
;     The chip layout is   c1 = right 
;                          c2 = left
;
;     Overscan region is read from fits header, trim section
;     is hardwired to [0:2031,0:2031].
;
; CALLING SEQUENCE:
;     ldss_stitch2, strct, objnum, im
;
; INPUTS:
;     inroot -- root name of image to be stiched (ccd1156c[1:4].fits)
;     imfinal -- name of output files file.
;
; OPTIONAL INPUTS:
;
;     inroot -- set image name by hand, ignores strct/objnum 
;
;     write_fits -- create output fits file.  Prompts for file name.
;
; OUTPUTS:
;     writes single combined file
;
; COMMENTS
;    trim section hardwired, assumes no binning
;
; MODIFICATION HISTORY:
;   MG 3/05
;   mg 8/05 modified from 4 -> 2 ccds
;- 
;--------------------------------------------------------------------
Pro ldss_stitch2, strct, indx, imfinal, write_fits=write_fits, hdr=hdr,$
                 outfile = outfile, inroot = inroot, PLOT=plot,gain=gain


;  if  N_params() LT 2  then begin 
;      print,'Syntax - ' + $
;        'ldss_stitch2, inroot, imfinal, /plot, /write_fits'
;      return
;  endif 


  ; GAINS ARE RELATIVE TO CHIP 1, DETERMINED FROM FLAT FIELDS
    if NOT keyword_set(gain) then gain=[1.0, 0.915]


  ; SET FILE
    if NOT keyword_set(inroot) then begin
       zr='000'
       if (strct[indx].ccdnum gt 9) then zr = '00'
       if (strct[indx].ccdnum gt 99) then zr = '0'
       if (strct[indx].ccdnum gt 999) then zr = ''
       inroot = 'rawdata/ccd'+ zr+string(strct[indx].ccdnum)                  
       inroot = strcompress(inroot,/remove_all)

       biasroot = strct[indx].bias_fil

    endif



  ; READ EACH CHIP, BIAS SUBTRACT, TRIM
    imfinal = fltarr(4064,4064)
    for i=1,2 do begin

        file = inroot + 'c' + string(i) + '.fits.gz'
        file =strcompress(file,/remove_all)

        print,'  Reading file ',file
        im   = mrdfits(file,0,hdr)

      ; BIAS SUBTRACT
        biasfile = biasroot + 'c' + string(i)+'.fits.gz'
        biasfile = strcompress(biasfile,/remove_all)
        bias = mrdfits(biasfile)
        im = im - bias


      ; OVERSCAN SUBTRACT
         print,'Overscan subtracting...'
         ov = djs_median(im[2032:2159,*],1)
         for j = 0,2031 do im[j,*] = im[j,*]-ov


      ; MULTIPLY BY GAIN, TRIM IMAGES
        im = im / gain[i-1]
        imtrim = im[0:2031,0:4063]

     ; COMBINE INTO SINGLE FILE
       if (i eq 2) then imfinal[0:2031,*]      = imtrim
       if (i eq 1) then imfinal[2032:4063,*]   = reverse(imtrim,1)

      

  endfor


; PLOT IF YOU WANT
  if keyword_set(plot) then atv,imfinal


; WRITE FITS OUTPUT, HEADER FROM CHIP 1
  if keyword_set(write_fits) then begin
          if NOT keyword_set(outfile) then begin
                 outfile = ' '
                 read,outfile,prompt='Enter output name: '
                 outfile = strcompress(outfile,/remove_all)
          endif

          mwrfits, 0, outfile, /create
          mwrfits, im, outfile, hdr
  endif


end
