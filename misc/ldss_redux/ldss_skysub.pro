;--------------------------------------------------------------------
;+
; NAME:
;     ldss_skysub
; 
; PURPOSE:
;    Sky subtract single image via bspline fitting 
;    to specified sky region
;
; CALLING SEQUENCE:
;    ldss_skysub, img, ivar, wav, sky
;
; INPUTS:
;   img   -- 2D processed image
;   ivar  -- 2D inverse variance
;   wav   -- 2D wavelength image
;
; OPTIONAL INPUTS
;    yobj -- yposition of object
;    bksp -- break spacing for bspline fit (angstrums)
;
;
; OUTPUTS:
;   sky -- sky subtracted image
;
;
; COMMENTS
;   could modify this to deal with bright, bleeding sky lines in chip2
;
;
; MODIFICATION HISTORY:
;  based on xidl/wfccd_subsky  mg 3/05
;- 
;--------------------------------------------------------------------
Pro ldss_skysub,im, ivar, wav, sky, yobj = yobj, bksp=bksp, plot=plot


  if  N_params() Lt 1  then begin
      print,'Syntax - ' + $
        'ldss_procobj, strct'
      return
  endif 

  ; SET SKY REGION
    if NOT keyword_set(yobj) then yobj = [40,230]

;yobj=[300,1000]  
  ; MAKE SKY MASK-- DON'T USE FLAGGED CR IN FIT
    msk = im
    msk[where(im ne 0)] = 1   

  ; MASK OUT OBJECT
    msk[*,yobj[0]:yobj[1]] = -100       

    ;msk[*,1100:*]=-100

  ; SET SKY PIXELS
    all_skypix = where(msk EQ 1)
    sv_wv = wav[all_skypix]
    sv_sky = im[all_skypix]
    sv_ivar= ivar[all_skypix]
    srt = sort(sv_wv)
    sky_wv = sv_wv[srt]
    sky_fx = sv_sky[srt]
    sky_ivar = sv_ivar[srt]
            

  ; SET AND SORT BKPTS
    wv1d = djs_median(wav[*,*],2)
    bkpts = [min(sky_wv)-1., wv1d, max(sky_wv) + 1.]


  ;  ADD BREAK POINTS NEAR BRIGHT SKY LINES  
;     qadd  = where(wv1d GT 6296 AND wv1d LE 6303)
;     addpt = wv1d(qadd) - (wv1d[1] - wv1d[0])/2.
;     bkpts = [bkpts, addpt]
;     qadd  = where(wv1d GT 6296 AND wv1d LE 6303)
;     addpt = wv1d(qadd) - (wv1d[1] - wv1d[0])/2.
;     bkpts = [bkpts, addpt]

    srt = sort(bkpts)   ; sort
    bkpts = bkpts[srt]


    
  ;  FIT BSPLINE TO SKY 
;     bset = bspline_iterfit(sky_wv, sky_fx,fullbkpt=bkpts)
     bset = bspline_iterfit(sky_wv, sky_fx,bkspace=0.5)


  ;  EVALUALTE BSPLINE AT 2D WAVELENGTH IMAGE
     skymodel = bspline_valu(wav, bset)


     if keyword_set(plot) then begin
;        plot,sky_wv,sky_fx,psym=3,xrange=[6290,6310]
        plot,sky_wv,sky_fx,psym=3,xrange=[6360,6370]
        bfit = bspline_valu(bset.fullbkpt, bset)
        loadct,12
        oplot,bset.fullbkpt,bfit,psym=1,color=100
        loadct,0     
    endif

    sky = im - skymodel


end
