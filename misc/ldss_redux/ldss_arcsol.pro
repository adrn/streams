;--------------------------------------------------------------------
;+
; NAME:
;     ldss_arcsol
; 
; PURPOSE:
;    Create 2D wavelength array for LDSS3 longslit.  
;    First run ldss_idarc to create rough arc solution.  
;    For every row, average 10 rows and fit arc lines. 
;
; CALLING SEQUENCE:
;    ldss_arcsol, strct,arcid
;
; INPUTS:
;    strct -- ldss structure
;    arcid -- index of arc frame in strct
;
; OUTPUTS:
;    Write 2D wavelength image to file Arc/arc_ccd***.fits
;
; OPTIONAL OUTPUTS:
;
;
; COMMENTS:
;    
;
; MODIFICATION HISTORY:
;   mg 3/05
;- 
;--------------------------------------------------------------------
Pro ldss_arcsol,strct, objnum, wavim, ytrim=ytrim,plot=plot, nccd4 = nccd4

  !p.multi=0
  if  N_params() Lt 2  then begin
      print,'Syntax - ' + $
        'ldss_arcsol, strct, arcid'
      return
  endif 

; SET Y-VALUES OVER WHICH TO SOLVE 
  if NOT keyword_set(ytrim) then ytrim = [1500,2700]


; PROCESS ARC FRAME
  arcfile = 'Arcs/arc_' + string(objnum) + '.fits'
  arcfile = strcompress(arcfile,/remove_all)

  arcid = where(strct.type eq 'ARC' and strct.objnum eq objnum,nflt)

  if keyword_set(nccd4) then ldss_stitch4,strct,arcid,arc else $
     ldss_stitch2,strct,arcid,arc
  readcol,'Arcs/arcid.dat',arc_pix,arc_lam
  atv,arc



  naxis1 = 4064
  naxis2 = ytrim[1]-ytrim[0] +1
; FOR EACH ROW, DETERMINE WAVELENGTH SOLUTION
      xpix = findgen(naxis1)
      wavim = fltarr(naxis1,naxis2)      ; 2d wavelength array
      navgcol = 10
      nfudge = 0
      for j=0,naxis2-1 do begin 

          n1 = j + ytrim[0] - navgcol   ; fit arc by averaging over 10 pix 
          n2 = j + ytrim[0] + navgcol   
          arc_col = total(arc[*,n1:n2],2)


         ; CENTRIOD EACH LINE BY FITTING A GAUSSIAN - could try
         ;    using kungifu peakiter routine
           fit_pix = arc_pix
           for i=0,n_elements(arc_pix) -1 do begin
               qfit = where(xpix ge arc_pix[i]-38 and xpix le arc_pix[i]+38)
               g=gaussfit(xpix[qfit],arc_col[qfit],a)
               fit_pix[i] = a(1)

;               plot,xpix[qfit],arc_col[qfit],title=arc_lam[i]
;               oplot,[fit_pix[i],fit_pix[i]],[0,1e4]

             ; IF FIT IS MESSED UP, REPLACE WITH ORIGINAL
               if (a(1) le 0) then fit_pix[i] = arc_pix[i]

           endfor 

         ; ARC SOLUTION -- FIT POLYNOMIAL TO ARC LINES
           nord=5
           afit = poly_fit(fit_pix, arc_lam,nord,yfit=fit_lam)

         ; REJECT WORSE LINE AND REFIT
           resd = avg(abs(arc_lam - fit_lam))
           tmp = max(abs(arc_lam - fit_lam),q)
           arc_lam[q]=-99
           q=where(arc_lam ne -99)
           arc_lam2=arc_lam[q]
           fit_pix2=fit_pix[q]
           afit = poly_fit(fit_pix2, arc_lam2,nord,yfit=fit_lam2)


         ; WRITE SOLUTION TO IMAGE
           wvl_col = 0.0
           for jj=0,nord do wvl_col = wvl_col + afit[jj] * xpix^jj

           ; PLOT RESIDUALS
             resd = avg(abs(arc_lam2 - fit_lam2))
             if (keyword_set(plot) and resd ge 0.1 and j ne 0) then begin
                plot,arc_lam2,arc_lam2-fit_lam2,psym=1, xtitle='Wavelength', ytitle='Residuals (AA)'
             endif

             wavim[*,j] = wvl_col

             xyouts,arc_lam2,arc_lam2-fit_lam2,arc_lam2
             print,(n1+n2)/2.,resd

         ; FIX OCCASSIONAL TRAIN WRECKS, MAXIMUM RESIDUAL SET = 0.1AA
           if (resd ge 0.1 and j ne 0) then begin
                wavim[*,j] = wavim[*,j-1]
                nfudge = nfudge + 1
                if (nfudge ge 25) then begin
                    tmp = max(abs(arc_lam2 - fit_lam2),q)
                    arc_lam2[q]=-99
                    q=where(arc_lam2 ne -99)
                    arc_lam3=arc_lam2[q]
                    fit_pix3=fit_pix2[q]
                    afit = poly_fit(fit_pix3, arc_lam3,nord,yfit=fit_lam3)
                    wvl_col = 0.0
                    for jj=0,nord do wvl_col = wvl_col + afit[jj] * xpix^jj

                    resd = avg(abs(arc_lam3 - fit_lam3))
                    if keyword_set(plot) then begin
                        plot,arc_lam3,arc_lam3-fit_lam3,psym=1, xtitle='Wavelength', $
                          ytitle='Residuals (AA)'
                        xyouts,arc_lam3,arc_lam3-fit_lam3,arc_lam3
                    endif

                    wavim[*,j] = wvl_col

                    print,(n1+n2)/2.,resd
                    if (resd ge 0.1 and j ne 0) then begin
                        print,'Arc solution residuals too large'
                    
                        return
                    endif
                endif 
            endif 


       endfor    
       mwrfits,wavim,arcfile,/create
       

end
