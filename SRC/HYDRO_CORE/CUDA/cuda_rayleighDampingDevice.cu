/* FastEddy®: SRC/HYDRO_CORE/CUDA/cuda_rayleighDampingDevice.cu 
* ©2016 University Corporation for Atmospheric Research
* 
* This file is licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
* 
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
/*---RAYLEIGH DAMPING LAYER*/
__constant__ int dampingLayerSelector_d;       // Rayleigh Damping Layer selector
__constant__ float dampingLayerDepth_d;       // Rayleigh Damping Layer Depth
/*---LATERAL RAYLEIGH DAMPING (SPONGE) LAYER*/
__constant__ int lateralDampingSelector_d;     // Lateral Rayleigh damping (sponge) selector: 0= off, 1= on
__constant__ int lateralDampingWidth_d;        // Lateral sponge width in cells, applied inward from each x/y face
__constant__ float lateralDampingCoeff_d;      // Lateral sponge maximum damping rate (1/s)

/*#################------------ RAYLEIGHDAMPING submodule function definitions ------------------#############*/
/*----->>>>> int cuda_rayleighDampingDeviceSetup();       ---------------------------------------------------------
 * Used to cudaMalloc and cudaMemcpy parameters and coordinate arrays, and for the RAYLEIGHDAMPING_CUDA submodule.
*/
extern "C" int cuda_rayleighDampingDeviceSetup(){
   int errorCode = CUDA_RAYLEIGHDAMPING_SUCCESS;

   cudaMemcpyToSymbol(dampingLayerSelector_d, &dampingLayerSelector, sizeof(int));
   cudaMemcpyToSymbol(dampingLayerDepth_d, &dampingLayerDepth, sizeof(float));
   cudaMemcpyToSymbol(lateralDampingSelector_d, &lateralDampingSelector, sizeof(int));
   cudaMemcpyToSymbol(lateralDampingWidth_d, &lateralDampingWidth, sizeof(int));
   cudaMemcpyToSymbol(lateralDampingCoeff_d, &lateralDampingCoeff, sizeof(float));

   return(errorCode);
} //end cuda_rayleighDampingDeviceSetup()

/*----->>>>> extern "C" int cuda_rayleighDampingDeviceCleanup();  --------------------------------------------------
Used to free all malloced memory by the RAYLEIGHDAMPING submodule.
*/

extern "C" int cuda_rayleighDampingDeviceCleanup(){
   int errorCode = CUDA_RAYLEIGHDAMPING_SUCCESS;

   /* Free any RAYLEIGHDAMPING submodule arrays */

   return(errorCode);

}//end cuda_rayleighDampingDeviceCleanup()

/*----->>>>> __device__ void cudaDevice_topRayleighDampingLayerForcing();  ------------------------------------------
* Rayleigh damping layer forcing term 
*/
__device__ void cudaDevice_topRayleighDampingLayerForcing(float* scalarField, float* scalarFrhs,
                                                          float* rho, float* rhoBS, float* zPos_d){

  int i,j,k;
  int ijk,ijkTop;
  int iStride,jStride,kStride;
  float pi_o_2;
  float wBSval;

  pi_o_2 = 0.5*acos(-1.0);
  /*Establish necessary indices for spatial locality*/
  i = (blockIdx.x)*blockDim.x + threadIdx.x;
  j = (blockIdx.y)*blockDim.y + threadIdx.y;
  k = (blockIdx.z)*blockDim.z + threadIdx.z;
  iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
  jStride = (Nz_d+2*Nh_d);
  kStride = 1;
  ijk = i*iStride + j*jStride + k*kStride;
  ijkTop = i*iStride + j*jStride + (kMax_d-1)*kStride;
  if((i >= iMin_d-Nh_d)&&(i < iMax_d+Nh_d) &&
     (j >= jMin_d-Nh_d)&&(j < jMax_d+Nh_d) ){
     if(zPos_d[ijk] >= (zPos_d[ijkTop]-dampingLayerDepth_d)){
        cudaDevice_MomentumBS(W_INDX,zPos_d[ijk],&rhoBS[RHO_INDX_BS+ijk],&wBSval);
        scalarFrhs[ijk] = scalarFrhs[ijk]
                         -0.2*rho[ijk]*( pow(sinf(pi_o_2
                                    *(1.0-(zPos_d[ijkTop]-zPos_d[ijk])/dampingLayerDepth_d)) ,2) )
                             *(scalarField[ijk]/rho[ijk]-wBSval/rhoBS[RHO_INDX_BS+ijk]);
     }//endif zPos >= (ztop-z_d)
  }//end if k>=kMin_dh

} // end cudaDevice_topRayleighDampingLayerForcing

/*----->>>>> __device__ void cudaDevice_lateralRayleighDampingForcing();  -------------------------------------------
* Lateral Rayleigh damping (sponge) forcing term. Relaxes a prognostic momentum or potential-temperature field
* toward its geostrophic/base state within a sponge of lateralDampingWidth_d cells inward from each of the four
* lateral (x/y) domain faces. A sin^2 ramp gives zero damping at the interior edge of the sponge and maximum
* damping (lateralDampingCoeff_d) at the face; the x- and y-face ramps are combined by max() so corners take the
* stronger of the two. Direction-agnostic absorber: clean inflow on upwind faces, wake dissipation on downwind
* faces. Mirrors cudaDevice_topRayleighDampingLayerForcing but acts on lateral faces instead of the model top.
* fldIndx selects the relaxation target: U_INDX/V_INDX/W_INDX use the base-state momentum profile
* (cudaDevice_MomentumBS); THETA_INDX uses the base-state potential temperature field (thetaBS).
*/
__device__ void cudaDevice_lateralRayleighDampingForcing(int fldIndx, float* scalarField, float* scalarFrhs,
                                                         float* rho, float* rhoBS, float* thetaBS, float* zPos_d){

  int i,j,k;
  int ijk;
  int iStride,jStride,kStride;
  int gi,gj,Nxtot,Nytot,distX,distY;
  float pi_o_2;
  float rampX,rampY,ramp,fracX,fracY;
  float targetOverRho,momBSval;

  pi_o_2 = 0.5*acos(-1.0);
  /*Establish necessary indices for spatial locality*/
  i = (blockIdx.x)*blockDim.x + threadIdx.x;
  j = (blockIdx.y)*blockDim.y + threadIdx.y;
  k = (blockIdx.z)*blockDim.z + threadIdx.z;
  iStride = (Ny_d+2*Nh_d)*(Nz_d+2*Nh_d);
  jStride = (Nz_d+2*Nh_d);
  kStride = 1;
  if((i >= iMin_d)&&(i < iMax_d) &&
     (j >= jMin_d)&&(j < jMax_d) &&
     (k >= kMin_d)&&(k < kMax_d) ){
     ijk = i*iStride + j*jStride + k*kStride;
     /*Global 0-based horizontal indices (rank-safe; matches the cellpert convention: i-Nh + rankX*Nx)*/
     Nxtot = numProcsX_d*Nx_d;
     Nytot = numProcsY_d*Ny_d;
     gi = (i-Nh_d) + rankXid_d*Nx_d;
     gj = (j-Nh_d) + rankYid_d*Ny_d;
     /*Distance in cells to the nearest x-face (west=gi, east=Nxtot-1-gi) and y-face (south/north)*/
     distX = min(gi,(Nxtot-1)-gi);
     distY = min(gj,(Nytot-1)-gj);
     /*sin^2 ramp from interior edge (ramp=0) to face (ramp=1) for each direction; combine by max()*/
     rampX = 0.0;
     if(distX < lateralDampingWidth_d){
        fracX = 1.0 - ((float)distX)/((float)lateralDampingWidth_d);
        rampX = sinf(pi_o_2*fracX);
        rampX = rampX*rampX;
     }
     rampY = 0.0;
     if(distY < lateralDampingWidth_d){
        fracY = 1.0 - ((float)distY)/((float)lateralDampingWidth_d);
        rampY = sinf(pi_o_2*fracY);
        rampY = rampY*rampY;
     }
     ramp = fmaxf(rampX,rampY);
     if(ramp > 0.0){
        /*Relaxation target divided by density: base-state momentum profile, or base-state theta*/
        if(fldIndx==THETA_INDX){
           targetOverRho = thetaBS[ijk]/rhoBS[ijk];
        }else{
           cudaDevice_MomentumBS(fldIndx, zPos_d[ijk], &rhoBS[ijk], &momBSval);
           targetOverRho = momBSval/rhoBS[ijk];
        }
        scalarFrhs[ijk] = scalarFrhs[ijk]
                         -lateralDampingCoeff_d*ramp*rho[ijk]
                            *(scalarField[ijk]/rho[ijk]-targetOverRho);
     }//endif ramp>0
  }//end if non-halo cell

} // end cudaDevice_lateralRayleighDampingForcing

