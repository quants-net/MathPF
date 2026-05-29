/* Vendored from quants-net-analytics/pricer/main/commondefs/CommonMacro.h
 * (the QNSPACE namespace macro + DLLIXP DLL-export attribute).  Kept here so
 * MathPF's _kernels/ tree is self-contained and qna can re-vendor the kernels
 * with zero edits. */
#ifndef _COMMONMACRO_H_
#define _COMMONMACRO_H_

#define QNSPACE namespace QuantsNet

QNSPACE{

#if defined(_WIN32) || defined(_WIN64)

#ifdef  __DLLEXPORT_MODE__
#define DLLIXP __declspec( dllexport )
#else
#define DLLIXP __declspec( dllimport )
#endif

#else

#define DLLIXP

#endif

}

#endif
