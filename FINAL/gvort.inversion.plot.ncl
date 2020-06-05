;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; 
;
; Based on a fortran program by Prof. Gary Lackmann in NCSU for his SOR method 
; Appreciation for C. Jung, R. Miller, and D. White for their project in MEA class in NCSU
; Used by Prof. J.-H. Kim (jhkim99@snu.ac.kr) for his class in SNU in 2019
;
; This script plots geopotential heights and geostrophic vorticity, and
; inverts vorticity to get geopotential.
;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;===========================================================================
; README - Use only one solve method (sorSolve, jacSolve, nclSolve).  If
;        ---- no choice is used, script will plot orginal fields
;---------------------------------------------------------------------------
;        - Set plotInvertedKatrina to true if you want to see only Katrina
;        ---- inverted. Set plotInvertedNoKatrina to invert without Katrina
;---------------------------------------------------------------------------
;        - Script uses data passed from python API script to download ERAi
;        ---- data and then passed from grib->NETCDF converter
;---------------------------------------------------------------------------
;        - Figure will be named based on if you're plotting raw fields
;        ---- (e.g. geo.png), fields without Katrina (e.g. geo_NoKatrina),
;        ---- and the method used to solve (e.g. geo_NoKatrina_sor.png).
;===========================================================================

begin

;===========================================================================
; Set script options
;===========================================================================

; Script options
verbose               = True   ; print to terminal
; ; Solve methods   
sorSolve              = False  ; solve using SOR
jacSolve              = True   ; solve using Jacobian iteration
nclSolve              = False  ; solve using NCL inverse laplacian
; ; Plot options 
filterVort            = True   ; filter vorticity before plotting
plotGeo               = True   ; plot geopotential
plotVort              = True   ; plot vorticity
plotInvertedNoKatrina = False  ; plot inverted atm. sans Katrina
plotInvertedKatrina   = True   ; plot on inverted Katrina
plotBox               = True   ; plot box around the vorticity partition

; SOR/Jacobian options
itLimit       = .001           ; cut-off to check for convergence
numIterations = 1000000        ; max number of iterations
w             = 0.8            ; weight for SOR method

;===========================================================================
; Domain options
;===========================================================================

; Time to read (YYYYMMDDHH)
eventTime = 2005082812

; Latitude range
lat1 = -20.
lat2 = 70.

; Longitude range
lon1 = -150.
lon2 = -30.

; Level range
lev1 = 500.
lev2 = 500.

; Reference latitude and rotation rate of earth
phi0  = 30.
Omega = (2*3.14)/(24*60*60)
f0    = 2*Omega*sin(phi0*(3.14159/180.))

; Set map latitude range
plat1 = lat1+2.
plat2 = lat2-2.

; Set map longitude range
plon1 = lon1+2.
plon2 = lon2-2.

; Set map level
plev = 500.

;===========================================================================
; Set input and output directories
;===========================================================================

; Input 
idir  = "./"                              ; data directory
ifile = idir+"ERAi.6hour.0.75-deg.GEO.2005.nc"  ; data file

; Output
odir  = "./"                            ; image directory
geofile  = odir+"geo"                           ; image file to make
vortfile = odir+"vort"                          ; image file to make

;===========================================================================
; Read data
;===========================================================================

if verbose then
  print("Reading: "+ifile)
end if

; Add file
infile = addfile(ifile,"r")

; Read dimensions and get number of points in dimensions
lat = infile->lat
lon = infile->lon
nlat = dimsizes(lat)
nlon = dimsizes(lon)

; Read geopotential
geo = infile->GEO({eventTime},:,:)

; Find dx from lon
dlon = lon
dlon(1:nlon-2) = (lon(2:nlon-1)+lon(1:nlon-2))/2.-\ ; centered difference
                 (lon(0:nlon-3)+lon(1:nlon-2))/2.
dlon(0)        = lon(1)-lon(0)                      ; fwd difference
dlon(nlon-1)   = lon(nlon-1)-lon(nlon-2)            ; bkwd difference
dx = conform(geo,tofloat(dlon),1)                   ; set dx size to geo
dx = dx*conform(geo,tofloat(cos(lat*3.14159/180.)),0)*111000

; Find dy from lat
dlat = lat
dlat(1:nlat-2) = (lat(2:nlat-1)+lat(1:nlat-2))/2.-\ ; centered difference
                 (lat(0:nlat-3)+lat(1:nlat-2))/2.
dlat(0)        = lat(1)-lat(0)                      ; forward difference
dlat(nlat-1)   = lat(nlat-1)-lat(nlat-2)            ; backward difference
dy = conform(geo,tofloat(dlat),0)*111000            ; set dy size to geo

; Find ratio of dx and dy
gridRatio = dx/dy

;===========================================================================
; Calculate geostrophic vorticity
;===========================================================================

if verbose then
  print("Calculating geostropic vorticity")
end if

; Use lapsF to calculate fixed point Laplacian
vorticity = (1./f0)*lapsF(geo)
copy_VarCoords(geo,vorticity)

;===========================================================================
; Remove Katrina vorticity blob
;===========================================================================

; Set the vorticity threshold : select vort at 25N 60W
threshold = vorticity({25},{-60})

if verbose then
  print("Threshold Vorticity: "+threshold)
end if

; Select box for partitioning the vorticity
; ; Latitude band
klat1 = 18.
klat2 = 34.
; ; Longitude band
klon1 = -97.
klon2 = -79.

; Change vorticity names to remove ambiguity
oldVorticity = vorticity
newVorticity = vorticity
delete(vorticity)

; Set vorticity to threshold vorticity in the Katrina region
newVorticity({klat1:klat2},{klon1:klon2}) = where(\
               newVorticity({klat1:klat2},{klon1:klon2})\
               .ge.threshold,threshold,\
               newVorticity({klat1:klat2},{klon1:klon2}))
newVorticity({klat1:klat2},{klon1:klon2}) = where(\
               newVorticity({klat1:klat2},{klon1:klon2})\
               .le.threshold,threshold,\
               newVorticity({klat1:klat2},{klon1:klon2}))

; Set "vorticity" to whatever we want to plot
if plotInvertedNoKatrina then                 
  vorticity = newVorticity
  vorticity@long_name = "Vorticity: Without Katrina"
else if plotInvertedKatrina then
  vorticity = oldVorticity - newVorticity
  vorticity@long_name = "Vorticity: Katrina Only"
else
  vorticity = oldVorticity
  vorticity@long_name = "Vorticity: With Katrina"
end if end if
copy_VarCoords(oldVorticity,vorticity)
vorticity@units = "s-1"

;===========================================================================
; Invert newVorticity to get new geopotential using NCL ilapsF
;===========================================================================

if nclSolve then

  if verbose then
    print("Inverting vorticity using NCL ilapsf function...")
  end if

  ; Update newGeo with value from ilapsF, add domain average, otherwise 
  ; we get anomolous fields.
  newGeo = geo
  oldGeo = geo
  newGeo = ilapsF(vorticity*f0,0)
  if plotInvertedKatrina then
    newGeo = newGeo
  else
    newGeo = newGeo + avg(avg(geo))
  end if
  delete(geo)

end if

;===========================================================================
; Invert newVorticity for new geopotential using Jacobian interation
;===========================================================================

if jacSolve then

  if verbose then
    print("Inverting vorticity using iterative Jacobian method...")
  end if

  ; Set boundary condition and initial guess
  if plotInvertedKatrina then
    newGeo = geo*0.
    copy_VarCoords(geo,newGeo)
  else
    newGeo = geo
  end if
  oldGeo = geo
  delete(geo)

  ; Preallocate values for do loop
  check = 10000000.
  oldResid = 1000000.
  k = 0

  ; Perform Jacobi iteration
  do while(k.lt.numIterations)
    previousGeopotential = newGeo
    do i = 1,nlon-2
      do j = 1,nlat-2
        a1 = (1./(2.*(1+gridRatio(j,i)^2.)))
        a2 = newGeo(j,i+1) + newGeo(j,i-1)
        a3 = gridRatio(j,i)^2* \
             (newGeo(j+1,i)+newGeo(j-1,i)) 
        a4 = -(f0)*vorticity(j,i)*dx(j,i)^2
        newGeo(j,i) = a1*(a2 + a3 + a4)
      end do
    end do
    newResid = sqrt(avg(avg((newGeo-previousGeopotential)^2)))
    check = abs(newResid-oldResid)/oldResid
    oldResid = newResid
    if verbose .and. mod(k,10).eq.0 then
      print("  Iteration: "+k+", E = "+check)
    end if
    k = k+1
    if check.lt.itLimit then
      k = numIterations
    end if
  end do

end if


;===========================================================================
; Invert newVorticity for new geopotential using successive over-relaxation
;===========================================================================

if sorSolve then

  if verbose then
    print("Inverting vorticity using SOR method...")
  end if

  ; Set geopotential boundary conditions to 0 or to the original heights
  if plotInvertedKatrina then
    newGeo = geo*0.
    copy_VarCoords(geo,newGeo)
  else
    newGeo = geo
  end if
  oldGeo = geo
  delete(geo)
  
  ; Preallocate for do loop
  check = 10000000.
  oldResid = 1000000.
  k = 0

  ; Perform SOR iterations
  do while(k.lt.numIterations)
    previousGeopotential = newGeo
    do i = 1,nlon-2
      do j = 1,nlat-2
        a1 = (1./(2.*(1+gridRatio(j,i)^2.)))
        a2 = newGeo(j,i+1) + newGeo(j,i-1)
        a3 = gridRatio(j,i)^2* \
             (newGeo(j+1,i)+newGeo(j-1,i)) 
        a4 = -(f0)*vorticity(j,i)*dx(j,i)^2
        newGeo(j,i) = a1*(a2 + a3 + a4)*w \
                     + previousGeopotential(j,i)*(1.-w)
      end do
    end do
    newResid = sqrt(avg(avg((newGeo-previousGeopotential)^2)))
    check = abs(newResid-oldResid)/oldResid
    oldResid = newResid
    if verbose .and. mod(k,10).eq.0 then
      print("  Iteration: "+k+", E = "+check)
    end if
    k = k+1
    if check.lt.itLimit then
      k = numIterations
    end if
  end do
  
end if

;===========================================================================
; Change geopotential name based on what we want to plot
;===========================================================================

if plotInvertedNoKatrina then
  geo = newGeo
  geo@long_name = "Geopotential: Without Katrina"
else if plotInvertedKatrina then
  geo = newGeo
  geo@long_name = "Geopotential: Katrina Only"
else
  geo@long_name = "Geopotential: With Katrina"
end if end if

; Provide coordinates and units
copy_VarCoords(vorticity,geo)
geo@units = "m2 s-2"

;===========================================================================
; Filter vorticity to make it look nicer on the map
;===========================================================================

if filterVort then
  vorticity = wgt_runave_n_Wrap(vorticity,(/0.05,0.2,0.5,0.2,0.05/),0,0)
  vorticity = wgt_runave_n_Wrap(vorticity,(/0.05,0.2,0.5,0.2,0.05/),0,1)
end if

;===========================================================================
; Resources for plotting
;===========================================================================

; Geopotential resources
res                   = True                  ; plot mods desired
res@gsnFrame          = False                 ; wait to frame plot
res@gsnDraw           = False                 ; wait to draw plot
res@gsnAddCyclic      = False                 ; data isn't cyclic
res@cnFillOn          = True                  ; turn on color
res@cnLinesOn         = False                 ; no contour lines
res@cnLevelSelectionMode = "ManualLevels"     ; set manual contour levels
if plotInvertedKatrina then
  res@cnLevelSpacingF   = 200.                ; set contour spacing
  res@cnMaxLevelValF    = 200.                ; set max contour
else 
  res@cnLevelSpacingF   = 500.                ; set contour spacing
  res@cnMinLevelValF    = 50000.              ; set min contour
  res@cnMaxLevelValF    = 58000.              ; set max contour
end if
res@mpMinLonF         = plon1                 ; min longitude for map
res@mpMaxLonF         = plon2                 ; max longitude for map
res@mpMinLatF         = plat1                 ; min lat for map
res@mpMaxLatF         = plat2                 ; max lat for map
res@pmTickMarkDisplayMode = "Always"          ; display tickmarks
res@mpFillOn          = False                 ; don't fill continents
res@mpLabelsOn        = True                  ; map labels
res@mpPerimOn         = True                  ; map perimeter
res@mpGeophysicalLineThicknessF = 2.          ; map border thickness

; Vorticity specific resources
res2 = res                                    ; plot mods desired
res2@gsnFrame          = False                ; wait to frame plot
res2@gsnDraw           = False                ; wait to draw plot
res2@cnLineLabelsOn    = False                ; don't label contours
res2@cnFillOn          = True                 ; turn on color shading
res2@cnLinesOn         = False                ; no contour lines
res2@cnMinLevelValF    =  (-8.)*(10^(-4))     ; set min contour level
res2@cnMaxLevelValF    =  (8.)*(10^(-4))      ; set max contour level
res2@cnLevelSpacingF   =  (0.5)*(10^(-4))     ; set contour spacing
res2@mpGeophysicalLineThicknessF = 4.         ; map border thickness

; Geopotential overlay resources
res3                   = True                 ; plot mods desired
res3@gsnFrame          = False                ; wait to frame plot
res3@gsnDraw           = False                ; wait to draw plot
res3@gsnAddCyclic      = False                ; data isn't cyclic
res3@cnFillOn          = False                ; turn on color
res3@cnLinesOn         = True                 ; no contour lines
res3@cnLevelSelectionMode = "ManualLevels"    ; set manual contour levels
if plotInvertedKatrina then
  res3@cnLevelSpacingF   =  200.              ; set contour spacing
  res3@cnMaxLevelValF    =  200.              ; max contour
else
  res3@cnLevelSpacingF   =  500.              ; set contour spacing
  res3@cnMinLevelValF    = 45000.             ; min contour
end if
res3@cnLineLabelsOn        = True             ; label contours
res3@cnLineLabelDensityF   = 2.5              ; density of labels
res3@cnLineLabelPlacementMode = "Constant"    ; try to place consistently
res3@gsnContourNegLineDashPattern = 1         ; negative dashed contours

; Box plot resources
lres = True                                   ; plot mods desired
lres@xyLineThicknessF   = 6.                  ; line thickness
lres@xyDashPattern      = 1                   ; dashed lines
lres@gsnFrame = False                         ; wait to frame plot
lres@gsnDraw = False                          ; wait to draw plot

;===========================================================================
; Plot geopotential height
;===========================================================================

; Name file according to what we want to show
if plotInvertedNoKatrina then
  geofile = geofile+"_NoKatrina"
end if
if plotInvertedKatrina then
  geofile = geofile+"_Katrina"
end if

; Name fileaccording to how it was solved
if sorSolve then
  geofile = geofile+"_sor"
end if
if jacSolve then
  geofile = geofile+"_jac"
end if
if nclSolve then
  geofile = geofile+"_ncl"
end if

if plotGeo then
  
  if verbose then
    print("Plotting geopotential...")
  end if

  ; Set up the workspace and plot geo
  wks_type = "png"
  wks_type@wkHeight = 2000
  wks_type@wkWidth  = 2000
  wks  = gsn_open_wks(wks_type,geofile)
  gsn_define_colormap(wks,"BlGrYeOrReVi200")
  res@gsnRightString = tostring(eventTime)
  plot1 = gsn_csm_contour_map(wks,geo(:,:),res)

  ; plot the box over the partition region
  if plotBox then
    B_lon1 = klon1
    B_lon2 = klon2
    B_lat1 = klat1
    B_lat2 = klat2
    Bx = (/B_lon1,B_lon2,B_lon2,B_lon1,B_lon1/)
    By = (/B_lat1,B_lat1,B_lat2,B_lat2,B_lat1/)
    overlay(plot1,gsn_csm_xy(wks,Bx,By,lres))
  end if

  ; Create the plots
  draw(plot1)
  frame(wks)
  
  ; Trim excess whitespace of geo plot
  fig_file = geofile
  strver = systemfunc("ncl -V")
  ver = tofloat(strver)
  sub = str_sub_str(strver,ver+".","")
  if .not.isStrSubset(fig_file,".png") then
    fig_file = fig_file + ".png"
  end if
  cmd = "convert -trim "+fig_file+"   "+fig_file
;  if (ver.ge.6.2) .or. ((ver.eq.6.2).and.(sub.ge.1))
;  if fileexists(fig_file) then system(cmd) end if
;  else
;  if isfilepresent(fig_file) then system(cmd) end if
;  end if
  
end if

;===========================================================================
; Plot vorticity
;===========================================================================


; Name file according to what we want to show
if plotInvertedNoKatrina then
  vortfile = vortfile+"_NoKatrina"
end if
if plotInvertedKatrina then
  vortfile = vortfile+"_Katrina"
end if

; Name file according to how it was solved
if sorSolve then
  vortfile = vortfile+"_sor"
end if
if jacSolve then
  vortfile = vortfile+"_jac"
end if
if nclSolve then
  vortfile = vortfile+"_ncl"
end if

if plotVort then

  if verbose then
    print("Plotting vorticity...")
  end if

  ; Set up the workspace, plot vorticity, and overlay geopotential
  wks2  = gsn_open_wks(wks_type,vortfile)
  gsn_define_colormap(wks2,"BlueWhiteOrangeRed")
  res3@gsnLeftString  = ""
  res3@gsnRightString = ""
  res2@gsnRightString = eventTime
  plot2 = gsn_csm_contour_map(wks2,vorticity(:,:),res2)
  overlay(plot2,gsn_csm_contour(wks2,geo(:,:),res3))

  ; Plot box for partition area
  if plotBox then
    B_lon1 = klon1
    B_lon2 = klon2
    B_lat1 = klat1
    B_lat2 = klat2
    Bx = (/B_lon1,B_lon2,B_lon2,B_lon1,B_lon1/)
    By = (/B_lat1,B_lat1,B_lat2,B_lat2,B_lat1/)
    overlay(plot2,gsn_csm_xy(wks2,Bx,By,lres))
  end if

  ; Create plot
  draw(plot2)
  frame(wks2)

  ; Trim excess whitespace of geo plot
  fig_file = vortfile
  strver = systemfunc("ncl -V")
  ver = tofloat(strver)
  sub = str_sub_str(strver,ver+".","")
  if .not.isStrSubset(fig_file,".png") then
    fig_file = fig_file + ".png"
  end if
  cmd = "convert -trim "+fig_file+"   "+fig_file
;  if (ver.ge.6.2) .or. ((ver.eq.6.2).and.(sub.ge.1))
;  if fileexists(fig_file) then system(cmd) end if
;  else
;  if isfilepresent(fig_file) then system(cmd) end if
;  end if
  
end if

end
