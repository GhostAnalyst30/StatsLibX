(()=>{var a={};a.id=14,a.ids=[14],a.modules={261:a=>{"use strict";a.exports=require("next/dist/shared/lib/router/utils/app-paths")},846:a=>{"use strict";a.exports=require("next/dist/compiled/next-server/app-page.runtime.prod.js")},1025:a=>{"use strict";a.exports=require("next/dist/server/app-render/dynamic-access-async-storage.external.js")},1201:(a,b,c)=>{"use strict";c.r(b),c.d(b,{GlobalError:()=>E.a,__next_app__:()=>K,handler:()=>M,pages:()=>J,routeModule:()=>L,tree:()=>I});var d=c(9754),e=c(9117),f=c(6595),g=c(2324),h=c(9326),i=c(8928),j=c(175),k=c(12),l=c(4290),m=c(2696),n=c(2574),o=c(2802),p=c(7533),q=c(5229),r=c(2822),s=c(261),t=c(6453),u=c(2474),v=c(6713),w=c(1356),x=c(2685),y=c(6225),z=c(3446),A=c(2762),B=c(5742),C=c(6439),D=c(1170),E=c.n(D),F=c(2506),G=c(1203),H={};for(let a in F)0>["default","tree","pages","GlobalError","__next_app__","routeModule","handler"].indexOf(a)&&(H[a]=()=>F[a]);c.d(b,H);let I={children:["",{children:["docs",{children:["utils",{children:["__PAGE__",{},{page:[()=>Promise.resolve().then(c.bind(c,5660)),"C:\\Users\\Usuario\\Documents\\Emmanuel Ascendra\\Ciencia de Datos\\Librerias Python\\Libreria_Estadistica\\web\\src\\app\\docs\\utils\\page.tsx"]}]},{}]},{layout:[()=>Promise.resolve().then(c.bind(c,618)),"C:\\Users\\Usuario\\Documents\\Emmanuel Ascendra\\Ciencia de Datos\\Librerias Python\\Libreria_Estadistica\\web\\src\\app\\docs\\layout.tsx"]}]},{layout:[()=>Promise.resolve().then(c.bind(c,9553)),"C:\\Users\\Usuario\\Documents\\Emmanuel Ascendra\\Ciencia de Datos\\Librerias Python\\Libreria_Estadistica\\web\\src\\app\\layout.tsx"],"global-error":[()=>Promise.resolve().then(c.t.bind(c,1170,23)),"next/dist/client/components/builtin/global-error.js"],"not-found":[()=>Promise.resolve().then(c.t.bind(c,7028,23)),"next/dist/client/components/builtin/not-found.js"],forbidden:[()=>Promise.resolve().then(c.t.bind(c,461,23)),"next/dist/client/components/builtin/forbidden.js"],unauthorized:[()=>Promise.resolve().then(c.t.bind(c,2768,23)),"next/dist/client/components/builtin/unauthorized.js"]}]}.children,J=["C:\\Users\\Usuario\\Documents\\Emmanuel Ascendra\\Ciencia de Datos\\Librerias Python\\Libreria_Estadistica\\web\\src\\app\\docs\\utils\\page.tsx"],K={require:c,loadChunk:()=>Promise.resolve()},L=new d.AppPageRouteModule({definition:{kind:e.RouteKind.APP_PAGE,page:"/docs/utils/page",pathname:"/docs/utils",bundlePath:"",filename:"",appPaths:[]},userland:{loaderTree:I},distDir:".next",relativeProjectDir:""});async function M(a,b,d){var D;let H="/docs/utils/page";"/index"===H&&(H="/");let N=(0,h.getRequestMeta)(a,"postponed"),O=(0,h.getRequestMeta)(a,"minimalMode"),P=await L.prepare(a,b,{srcPage:H,multiZoneDraftMode:!1});if(!P)return b.statusCode=400,b.end("Bad Request"),null==d.waitUntil||d.waitUntil.call(d,Promise.resolve()),null;let{buildId:Q,query:R,params:S,parsedUrl:T,pageIsDynamic:U,buildManifest:V,nextFontManifest:W,reactLoadableManifest:X,serverActionsManifest:Y,clientReferenceManifest:Z,subresourceIntegrityManifest:$,prerenderManifest:_,isDraftMode:aa,resolvedPathname:ab,revalidateOnlyGenerated:ac,routerServerContext:ad,nextConfig:ae,interceptionRoutePatterns:af}=P,ag=T.pathname||"/",ah=(0,s.normalizeAppPath)(H),{isOnDemandRevalidate:ai}=P,aj=L.match(ag,_),ak=!!_.routes[ab],al=!!(aj||ak||_.routes[ah]),am=a.headers["user-agent"]||"",an=(0,v.getBotType)(am),ao=(0,q.isHtmlBotRequest)(a),ap=(0,h.getRequestMeta)(a,"isPrefetchRSCRequest")??"1"===a.headers[u.NEXT_ROUTER_PREFETCH_HEADER],aq=(0,h.getRequestMeta)(a,"isRSCRequest")??(0,n.f)(a.headers[u.RSC_HEADER]),ar=(0,t.getIsPossibleServerAction)(a),as=(0,m.checkIsAppPPREnabled)(ae.experimental.ppr)&&(null==(D=_.routes[ah]??_.dynamicRoutes[ah])?void 0:D.renderingMode)==="PARTIALLY_STATIC",at=!1,au=!1,av=as?N:void 0,aw=as&&aq&&!ap,ax=(0,h.getRequestMeta)(a,"segmentPrefetchRSCRequest"),ay=!am||(0,q.shouldServeStreamingMetadata)(am,ae.htmlLimitedBots);ao&&as&&(al=!1,ay=!1);let az=!0===L.isDev||!al||"string"==typeof N||aw,aA=ao&&as,aB=null;aa||!al||az||ar||av||aw||(aB=ab);let aC=aB;!aC&&L.isDev&&(aC=ab),L.isDev||aa||!al||!aq||aw||(0,k.d)(a.headers);let aD={...F,tree:I,pages:J,GlobalError:E(),handler:M,routeModule:L,__next_app__:K};Y&&Z&&(0,p.setReferenceManifestsSingleton)({page:H,clientReferenceManifest:Z,serverActionsManifest:Y,serverModuleMap:(0,r.createServerModuleMap)({serverActionsManifest:Y})});let aE=a.method||"GET",aF=(0,g.getTracer)(),aG=aF.getActiveScopeSpan();try{let f=L.getVaryHeader(ab,af);b.setHeader("Vary",f);let k=async(c,d)=>{let e=new l.NodeNextRequest(a),f=new l.NodeNextResponse(b);return L.render(e,f,d).finally(()=>{if(!c)return;c.setAttributes({"http.status_code":b.statusCode,"next.rsc":!1});let d=aF.getRootSpanAttributes();if(!d)return;if(d.get("next.span_type")!==i.BaseServerSpan.handleRequest)return void console.warn(`Unexpected root span type '${d.get("next.span_type")}'. Please report this Next.js issue https://github.com/vercel/next.js`);let e=d.get("next.route");if(e){let a=`${aE} ${e}`;c.setAttributes({"next.route":e,"http.route":e,"next.span_name":a}),c.updateName(a)}else c.updateName(`${aE} ${a.url}`)})},m=async({span:e,postponed:f,fallbackRouteParams:g})=>{let i={query:R,params:S,page:ah,sharedContext:{buildId:Q},serverComponentsHmrCache:(0,h.getRequestMeta)(a,"serverComponentsHmrCache"),fallbackRouteParams:g,renderOpts:{App:()=>null,Document:()=>null,pageConfig:{},ComponentMod:aD,Component:(0,j.T)(aD),params:S,routeModule:L,page:H,postponed:f,shouldWaitOnAllReady:aA,serveStreamingMetadata:ay,supportsDynamicResponse:"string"==typeof f||az,buildManifest:V,nextFontManifest:W,reactLoadableManifest:X,subresourceIntegrityManifest:$,serverActionsManifest:Y,clientReferenceManifest:Z,setIsrStatus:null==ad?void 0:ad.setIsrStatus,dir:c(9902).join(process.cwd(),L.relativeProjectDir),isDraftMode:aa,isRevalidate:al&&!f&&!aw,botType:an,isOnDemandRevalidate:ai,isPossibleServerAction:ar,assetPrefix:ae.assetPrefix,nextConfigOutput:ae.output,crossOrigin:ae.crossOrigin,trailingSlash:ae.trailingSlash,previewProps:_.preview,deploymentId:ae.deploymentId,enableTainting:ae.experimental.taint,htmlLimitedBots:ae.htmlLimitedBots,devtoolSegmentExplorer:ae.experimental.devtoolSegmentExplorer,reactMaxHeadersLength:ae.reactMaxHeadersLength,multiZoneDraftMode:!1,incrementalCache:(0,h.getRequestMeta)(a,"incrementalCache"),cacheLifeProfiles:ae.experimental.cacheLife,basePath:ae.basePath,serverActions:ae.experimental.serverActions,...at?{nextExport:!0,supportsDynamicResponse:!1,isStaticGeneration:!0,isRevalidate:!0,isDebugDynamicAccesses:at}:{},experimental:{isRoutePPREnabled:as,expireTime:ae.expireTime,staleTimes:ae.experimental.staleTimes,cacheComponents:!!ae.experimental.cacheComponents,clientSegmentCache:!!ae.experimental.clientSegmentCache,clientParamParsing:!!ae.experimental.clientParamParsing,dynamicOnHover:!!ae.experimental.dynamicOnHover,inlineCss:!!ae.experimental.inlineCss,authInterrupts:!!ae.experimental.authInterrupts,clientTraceMetadata:ae.experimental.clientTraceMetadata||[]},waitUntil:d.waitUntil,onClose:a=>{b.on("close",a)},onAfterTaskError:()=>{},onInstrumentationRequestError:(b,c,d)=>L.onRequestError(a,b,d,ad),err:(0,h.getRequestMeta)(a,"invokeError"),dev:L.isDev}},l=await k(e,i),{metadata:m}=l,{cacheControl:n,headers:o={},fetchTags:p}=m;if(p&&(o[z.NEXT_CACHE_TAGS_HEADER]=p),a.fetchMetrics=m.fetchMetrics,al&&(null==n?void 0:n.revalidate)===0&&!L.isDev&&!as){let a=m.staticBailoutInfo,b=Object.defineProperty(Error(`Page changed from static to dynamic at runtime ${ab}${(null==a?void 0:a.description)?`, reason: ${a.description}`:""}
see more here https://nextjs.org/docs/messages/app-static-to-dynamic-error`),"__NEXT_ERROR_CODE",{value:"E132",enumerable:!1,configurable:!0});if(null==a?void 0:a.stack){let c=a.stack;b.stack=b.message+c.substring(c.indexOf("\n"))}throw b}return{value:{kind:w.CachedRouteKind.APP_PAGE,html:l,headers:o,rscData:m.flightData,postponed:m.postponed,status:m.statusCode,segmentData:m.segmentData},cacheControl:n}},n=async({hasResolved:c,previousCacheEntry:f,isRevalidating:g,span:i})=>{let j,k=!1===L.isDev,l=c||b.writableEnded;if(ai&&ac&&!f&&!O)return(null==ad?void 0:ad.render404)?await ad.render404(a,b):(b.statusCode=404,b.end("This page could not be found")),null;if(aj&&(j=(0,x.parseFallbackField)(aj.fallback)),j===x.FallbackMode.PRERENDER&&(0,v.isBot)(am)&&(!as||ao)&&(j=x.FallbackMode.BLOCKING_STATIC_RENDER),(null==f?void 0:f.isStale)===-1&&(ai=!0),ai&&(j!==x.FallbackMode.NOT_FOUND||f)&&(j=x.FallbackMode.BLOCKING_STATIC_RENDER),!O&&j!==x.FallbackMode.BLOCKING_STATIC_RENDER&&aC&&!l&&!aa&&U&&(k||!ak)){let b;if((k||aj)&&j===x.FallbackMode.NOT_FOUND)throw new C.NoFallbackError;if(as&&!aq){let c="string"==typeof(null==aj?void 0:aj.fallback)?aj.fallback:k?ah:null;if(b=await L.handleResponse({cacheKey:c,req:a,nextConfig:ae,routeKind:e.RouteKind.APP_PAGE,isFallback:!0,prerenderManifest:_,isRoutePPREnabled:as,responseGenerator:async()=>m({span:i,postponed:void 0,fallbackRouteParams:k||au?(0,o.u)(ah):null}),waitUntil:d.waitUntil}),null===b)return null;if(b)return delete b.cacheControl,b}}let n=ai||g||!av?void 0:av;if(at&&void 0!==n)return{cacheControl:{revalidate:1,expire:void 0},value:{kind:w.CachedRouteKind.PAGES,html:y.default.EMPTY,pageData:{},headers:void 0,status:void 0}};let p=U&&as&&((0,h.getRequestMeta)(a,"renderFallbackShell")||au)?(0,o.u)(ag):null;return m({span:i,postponed:n,fallbackRouteParams:p})},p=async c=>{var f,g,i,j,k;let l,o=await L.handleResponse({cacheKey:aB,responseGenerator:a=>n({span:c,...a}),routeKind:e.RouteKind.APP_PAGE,isOnDemandRevalidate:ai,isRoutePPREnabled:as,req:a,nextConfig:ae,prerenderManifest:_,waitUntil:d.waitUntil});if(aa&&b.setHeader("Cache-Control","private, no-cache, no-store, max-age=0, must-revalidate"),L.isDev&&b.setHeader("Cache-Control","no-store, must-revalidate"),!o){if(aB)throw Object.defineProperty(Error("invariant: cache entry required but not generated"),"__NEXT_ERROR_CODE",{value:"E62",enumerable:!1,configurable:!0});return null}if((null==(f=o.value)?void 0:f.kind)!==w.CachedRouteKind.APP_PAGE)throw Object.defineProperty(Error(`Invariant app-page handler received invalid cache entry ${null==(i=o.value)?void 0:i.kind}`),"__NEXT_ERROR_CODE",{value:"E707",enumerable:!1,configurable:!0});let p="string"==typeof o.value.postponed;al&&!aw&&(!p||ap)&&(O||b.setHeader("x-nextjs-cache",ai?"REVALIDATED":o.isMiss?"MISS":o.isStale?"STALE":"HIT"),b.setHeader(u.NEXT_IS_PRERENDER_HEADER,"1"));let{value:q}=o;if(av)l={revalidate:0,expire:void 0};else if(O&&aq&&!ap&&as)l={revalidate:0,expire:void 0};else if(!L.isDev)if(aa)l={revalidate:0,expire:void 0};else if(al){if(o.cacheControl)if("number"==typeof o.cacheControl.revalidate){if(o.cacheControl.revalidate<1)throw Object.defineProperty(Error(`Invalid revalidate configuration provided: ${o.cacheControl.revalidate} < 1`),"__NEXT_ERROR_CODE",{value:"E22",enumerable:!1,configurable:!0});l={revalidate:o.cacheControl.revalidate,expire:(null==(j=o.cacheControl)?void 0:j.expire)??ae.expireTime}}else l={revalidate:z.CACHE_ONE_YEAR,expire:void 0}}else b.getHeader("Cache-Control")||(l={revalidate:0,expire:void 0});if(o.cacheControl=l,"string"==typeof ax&&(null==q?void 0:q.kind)===w.CachedRouteKind.APP_PAGE&&q.segmentData){b.setHeader(u.NEXT_DID_POSTPONE_HEADER,"2");let c=null==(k=q.headers)?void 0:k[z.NEXT_CACHE_TAGS_HEADER];O&&al&&c&&"string"==typeof c&&b.setHeader(z.NEXT_CACHE_TAGS_HEADER,c);let d=q.segmentData.get(ax);return void 0!==d?(0,B.sendRenderResult)({req:a,res:b,generateEtags:ae.generateEtags,poweredByHeader:ae.poweredByHeader,result:y.default.fromStatic(d,u.RSC_CONTENT_TYPE_HEADER),cacheControl:o.cacheControl}):(b.statusCode=204,(0,B.sendRenderResult)({req:a,res:b,generateEtags:ae.generateEtags,poweredByHeader:ae.poweredByHeader,result:y.default.EMPTY,cacheControl:o.cacheControl}))}let r=(0,h.getRequestMeta)(a,"onCacheEntry");if(r&&await r({...o,value:{...o.value,kind:"PAGE"}},{url:(0,h.getRequestMeta)(a,"initURL")}))return null;if(p&&av)throw Object.defineProperty(Error("Invariant: postponed state should not be present on a resume request"),"__NEXT_ERROR_CODE",{value:"E396",enumerable:!1,configurable:!0});if(q.headers){let a={...q.headers};for(let[c,d]of(O&&al||delete a[z.NEXT_CACHE_TAGS_HEADER],Object.entries(a)))if(void 0!==d)if(Array.isArray(d))for(let a of d)b.appendHeader(c,a);else"number"==typeof d&&(d=d.toString()),b.appendHeader(c,d)}let s=null==(g=q.headers)?void 0:g[z.NEXT_CACHE_TAGS_HEADER];if(O&&al&&s&&"string"==typeof s&&b.setHeader(z.NEXT_CACHE_TAGS_HEADER,s),!q.status||aq&&as||(b.statusCode=q.status),!O&&q.status&&G.RedirectStatusCode[q.status]&&aq&&(b.statusCode=200),p&&b.setHeader(u.NEXT_DID_POSTPONE_HEADER,"1"),aq&&!aa){if(void 0===q.rscData){if(q.postponed)throw Object.defineProperty(Error("Invariant: Expected postponed to be undefined"),"__NEXT_ERROR_CODE",{value:"E372",enumerable:!1,configurable:!0});return(0,B.sendRenderResult)({req:a,res:b,generateEtags:ae.generateEtags,poweredByHeader:ae.poweredByHeader,result:q.html,cacheControl:aw?{revalidate:0,expire:void 0}:o.cacheControl})}return(0,B.sendRenderResult)({req:a,res:b,generateEtags:ae.generateEtags,poweredByHeader:ae.poweredByHeader,result:y.default.fromStatic(q.rscData,u.RSC_CONTENT_TYPE_HEADER),cacheControl:o.cacheControl})}let t=q.html;if(!p||O||aq)return(0,B.sendRenderResult)({req:a,res:b,generateEtags:ae.generateEtags,poweredByHeader:ae.poweredByHeader,result:t,cacheControl:o.cacheControl});if(at)return t.push(new ReadableStream({start(a){a.enqueue(A.ENCODED_TAGS.CLOSED.BODY_AND_HTML),a.close()}})),(0,B.sendRenderResult)({req:a,res:b,generateEtags:ae.generateEtags,poweredByHeader:ae.poweredByHeader,result:t,cacheControl:{revalidate:0,expire:void 0}});let v=new TransformStream;return t.push(v.readable),m({span:c,postponed:q.postponed,fallbackRouteParams:null}).then(async a=>{var b,c;if(!a)throw Object.defineProperty(Error("Invariant: expected a result to be returned"),"__NEXT_ERROR_CODE",{value:"E463",enumerable:!1,configurable:!0});if((null==(b=a.value)?void 0:b.kind)!==w.CachedRouteKind.APP_PAGE)throw Object.defineProperty(Error(`Invariant: expected a page response, got ${null==(c=a.value)?void 0:c.kind}`),"__NEXT_ERROR_CODE",{value:"E305",enumerable:!1,configurable:!0});await a.value.html.pipeTo(v.writable)}).catch(a=>{v.writable.abort(a).catch(a=>{console.error("couldn't abort transformer",a)})}),(0,B.sendRenderResult)({req:a,res:b,generateEtags:ae.generateEtags,poweredByHeader:ae.poweredByHeader,result:t,cacheControl:{revalidate:0,expire:void 0}})};if(!aG)return await aF.withPropagatedContext(a.headers,()=>aF.trace(i.BaseServerSpan.handleRequest,{spanName:`${aE} ${a.url}`,kind:g.SpanKind.SERVER,attributes:{"http.method":aE,"http.target":a.url}},p));await p(aG)}catch(b){throw b instanceof C.NoFallbackError||await L.onRequestError(a,b,{routerKind:"App Router",routePath:H,routeType:"render",revalidateReason:(0,f.c)({isRevalidate:al,isOnDemandRevalidate:ai})},ad),b}}},3033:a=>{"use strict";a.exports=require("next/dist/server/app-render/work-unit-async-storage.external.js")},3164:(a,b,c)=>{Promise.resolve().then(c.bind(c,7870)),Promise.resolve().then(c.bind(c,6634))},3295:a=>{"use strict";a.exports=require("next/dist/server/app-render/after-task-async-storage.external.js")},3911:(a,b,c)=>{Promise.resolve().then(c.bind(c,6279)),Promise.resolve().then(c.bind(c,341))},5660:(a,b,c)=>{"use strict";c.r(b),c.d(b,{default:()=>h});var d=c(5338);let e=(0,c(9740).A)("wrench",[["path",{d:"M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.106-3.105c.32-.322.863-.22.983.218a6 6 0 0 1-8.259 7.057l-7.91 7.91a1 1 0 0 1-2.999-3l7.91-7.91a6 6 0 0 1 7.057-8.259c.438.12.54.662.219.984z",key:"1ngwbx"}]]);var f=c(8694),g=c(6634);function h(){return(0,d.jsxs)(d.Fragment,{children:[(0,d.jsx)(f.w,{title:"UtilsStats",description:"A utility class providing helper functions for data loading, validation, formatting, statistical testing, outlier detection, effect size calculation, and visualisation configuration. Complements the core statistical classes with practical data science utilities.",icon:(0,d.jsx)(e,{className:"w-6 h-6"}),version:"0.2.8"}),(0,d.jsxs)("section",{className:"mb-12",children:[(0,d.jsx)("h2",{className:"section-title",children:"Class Overview"}),(0,d.jsxs)("p",{className:"text-sm text-muted leading-relaxed",children:["The ",(0,d.jsx)("code",{className:"code-inline",children:"UtilsStats"})," class provides a collection of standalone utility methods for common data science workflows. It includes functions for loading data from various file formats, validating and converting data, formatting numbers, performing normality tests, calculating confidence intervals, detecting outliers, computing effect sizes, and generating publication-ready plots."]})]}),(0,d.jsxs)("section",{className:"mb-12",children:[(0,d.jsx)("h2",{className:"section-title",children:"Configuration Methods"}),(0,d.jsx)("p",{className:"text-sm text-muted leading-relaxed mb-4",children:"These methods control the global behaviour of plotting and output settings used across the visualisation utilities."}),(0,d.jsxs)("div",{className:"method-list",children:[(0,d.jsx)(g.MethodCard,{name:"set_plot_backend",signature:"set_plot_backend(backend: Literal['matplotlib', 'seaborn', 'plotly']) -> None",description:"Set the default visualisation backend for all plotting methods. This determines which library is used to render charts and figures.",parameters:[{name:"backend",type:"'matplotlib' | 'seaborn' | 'plotly'",description:"Name of the plotting backend to use globally."}],returns:"None",example:`from stats_lib import UtilsStats

utils = UtilsStats()

# Use Plotly for interactive charts
utils.set_plot_backend("plotly")

# Use Seaborn for statistical visualisations
utils.set_plot_backend("seaborn")`}),(0,d.jsx)(g.MethodCard,{name:"set_default_figsize",signature:"set_default_figsize(figsize: tuple[int, int]) -> None",description:"Set the default figure size (width, height) in inches for all plots.",parameters:[{name:"figsize",type:"tuple[int, int]",description:"Figure dimensions as (width, height) in inches."}],returns:"None",example:`from stats_lib import UtilsStats

utils = UtilsStats()

# Set default figure size to 12x6 inches
utils.set_default_figsize((12, 6))`}),(0,d.jsx)(g.MethodCard,{name:"set_save_fig_options",signature:"set_save_fig_options(save_fig: bool, fig_format: str = 'png', fig_dpi: int = 300, figures_dir: str = 'figures') -> None",description:"Configure whether plots are automatically saved to disk, the image format, resolution, and output directory.",parameters:[{name:"save_fig",type:"bool",description:"Whether to automatically save figures to disk."},{name:"fig_format",type:"str",description:"Image file format (e.g. 'png', 'pdf', 'svg', 'jpg').",default:"'png'"},{name:"fig_dpi",type:"int",description:"Resolution of saved figures in DPI.",default:"300"},{name:"figures_dir",type:"str",description:"Directory path where figures will be saved.",default:"'figures'"}],returns:"None",example:`from stats_lib import UtilsStats

utils = UtilsStats()

# Automatically save plots as high-res PNGs
utils.set_save_fig_options(
    save_fig=True,
    fig_format="png",
    fig_dpi=300,
    figures_dir="./output/figures"
)`})]})]}),(0,d.jsxs)("section",{className:"mb-12",children:[(0,d.jsx)("h2",{className:"section-title",children:"Data Loading & Validation"}),(0,d.jsxs)("div",{className:"method-list",children:[(0,d.jsx)(g.MethodCard,{name:"load_data",signature:"load_data(path: str, **kwargs: Any) -> pd.DataFrame",description:"Load data from a file into a pandas DataFrame. Supports CSV, Excel (.xls, .xlsx), JSON, Parquet, and Feather formats. File type is inferred from the extension.",parameters:[{name:"path",type:"str",description:"Path to the data file."},{name:"**kwargs",type:"Any",description:"Additional keyword arguments passed to the underlying pandas reader (e.g. sep, header, sheet_name)."}],returns:"pd.DataFrame",example:`from stats_lib import UtilsStats

utils = UtilsStats()

# Load CSV
df_csv = utils.load_data("data.csv")

# Load Excel with specific sheet
df_excel = utils.load_data("data.xlsx", sheet_name="Sheet1")

# Load CSV with custom separator
df_tsv = utils.load_data("data.tsv", sep="\\t")

# Load JSON
df_json = utils.load_data("data.json")`}),(0,d.jsx)(g.MethodCard,{name:"validate_dataframe",signature:"validate_dataframe(data: pd.DataFrame | np.ndarray | list[list] | dict) -> pd.DataFrame",description:"Validate and convert input data into a pandas DataFrame. Accepts DataFrames, numpy arrays, lists of lists, or dictionaries. Raises descriptive errors for empty or invalid inputs.",parameters:[{name:"data",type:"pd.DataFrame | np.ndarray | list[list] | dict",description:"Input data in any supported format."}],returns:"pd.DataFrame",example:`import numpy as np
from stats_lib import UtilsStats

utils = UtilsStats()

# From numpy array
arr = np.array([[1, 2], [3, 4], [5, 6]])
df = utils.validate_dataframe(arr)

# From list of lists
data = [[1, "A"], [2, "B"], [3, "C"]]
df = utils.validate_dataframe(data)

# From dict
data = {"x": [1, 2, 3], "y": [4, 5, 6]}
df = utils.validate_dataframe(data)`})]})]}),(0,d.jsxs)("section",{className:"mb-12",children:[(0,d.jsx)("h2",{className:"section-title",children:"Formatting"}),(0,d.jsx)("div",{className:"method-list",children:(0,d.jsx)(g.MethodCard,{name:"format_number",signature:"format_number(num: float | int, decimals: int = 6, scientific: bool = False) -> str",description:"Format a numeric value as a string with a specified number of decimal places. Optionally use scientific notation for very large or small numbers.",parameters:[{name:"num",type:"float | int",description:"The numeric value to format."},{name:"decimals",type:"int",description:"Number of decimal places to display.",default:"6"},{name:"scientific",type:"bool",description:"Whether to use scientific notation.",default:"False"}],returns:"str",example:`from stats_lib import UtilsStats

utils = UtilsStats()

# Standard formatting
print(utils.format_number(3.1415926535, decimals=2))  # "3.14"

# Scientific notation
print(utils.format_number(0.00000123, scientific=True))  # "1.23e-06"

# Default decimal places
print(utils.format_number(1234.56789))  # "1234.567890"`})})]}),(0,d.jsxs)("section",{className:"mb-12",children:[(0,d.jsx)("h2",{className:"section-title",children:"Statistical Tests"}),(0,d.jsxs)("div",{className:"method-list",children:[(0,d.jsx)(g.MethodCard,{name:"check_normality",signature:"check_normality(data: pd.DataFrame, column: str | None = None, alpha: float = 0.05) -> dict",description:"Perform the Shapiro-Wilk normality test on a dataset or a specific column. Returns a dictionary with the test statistic, p-value, a boolean indicating whether the data is normally distributed at the given significance level, and a message.",parameters:[{name:"data",type:"pd.DataFrame",description:"Input DataFrame containing the data to test."},{name:"column",type:"str | None",description:"Column name to test. If None, tests all numeric columns.",default:"None"},{name:"alpha",type:"float",description:"Significance level for the hypothesis test.",default:"0.05"}],returns:"dict",example:`import pandas as pd
from stats_lib import UtilsStats

utils = UtilsStats()

df = pd.DataFrame({"values": [2.1, 2.5, 2.8, 3.0, 3.2, 3.5, 3.8, 4.0, 4.2, 4.5]})

result = utils.check_normality(df, column="values", alpha=0.05)
print(result)
# {'statistic': 0.987, 'p_value': 0.952, 'is_normal': True,
#  'message': 'Data appears normally distributed (p=0.952, alpha=0.05)'}`}),(0,d.jsx)(g.MethodCard,{name:"calculate_confidence_intervals",signature:"calculate_confidence_intervals(data: pd.DataFrame, column: str | None = None, confidence_level: float = 0.95, method: Literal['parametric', 'bootstrap'] = 'parametric') -> dict",description:"Calculate confidence intervals for the mean of a dataset or specific column. Supports parametric (normal-based) and bootstrap methods.",parameters:[{name:"data",type:"pd.DataFrame",description:"Input DataFrame."},{name:"column",type:"str | None",description:"Column name. If None, computes CI for all numeric columns.",default:"None"},{name:"confidence_level",type:"float",description:"Confidence level (between 0 and 1).",default:"0.95"},{name:"method",type:"'parametric' | 'bootstrap'",description:"Method for CI calculation. 'parametric' uses the normal distribution; 'bootstrap' uses resampling.",default:"'parametric'"}],returns:"dict",example:`import pandas as pd
from stats_lib import UtilsStats

utils = UtilsStats()

df = pd.DataFrame({"score": [85, 92, 78, 95, 88, 76, 91, 84, 90, 79]})

# Parametric CI
ci = utils.calculate_confidence_intervals(df, column="score", confidence_level=0.95)
print(ci)
# {'column': 'score', 'mean': 85.8, 'ci_lower': 81.23, 'ci_upper': 90.37,
#  'confidence_level': 0.95, 'method': 'parametric'}

# Bootstrap CI
ci_boot = utils.calculate_confidence_intervals(
    df, column="score", confidence_level=0.95, method="bootstrap"
)
print(ci_boot)`})]})]}),(0,d.jsxs)("section",{className:"mb-12",children:[(0,d.jsx)("h2",{className:"section-title",children:"Outlier Detection"}),(0,d.jsx)("div",{className:"method-list",children:(0,d.jsx)(g.MethodCard,{name:"detect_outliers",signature:"detect_outliers(data: pd.DataFrame, column: str | None = None, method: Literal['iqr', 'zscore', 'isolation_forest'] = 'iqr', **kwargs: Any) -> pd.Series",description:"Detect outliers in a dataset using IQR, z-score, or Isolation Forest methods. Returns a boolean Series where True indicates an outlier.",parameters:[{name:"data",type:"pd.DataFrame",description:"Input DataFrame."},{name:"column",type:"str | None",description:"Column name to analyse. If None, detects outliers across all numeric columns.",default:"None"},{name:"method",type:"'iqr' | 'zscore' | 'isolation_forest'",description:"Detection algorithm. 'iqr' uses the interquartile range rule; 'zscore' uses standardised scores; 'isolation_forest' uses an ensemble of isolation trees.",default:"'iqr'"},{name:"**kwargs",type:"Any",description:"Additional keyword arguments passed to the detection method (e.g. threshold for IQR/zscore, contamination for isolation_forest)."}],returns:"pd.Series (boolean mask)",example:`import pandas as pd
from stats_lib import UtilsStats

utils = UtilsStats()

df = pd.DataFrame({"value": [10, 12, 11, 13, 100, 11, 12, 9, 14, 150]})

# IQR method
outliers_iqr = utils.detect_outliers(df, column="value", method="iqr")
print(df[outliers_iqr])

# Z-score method with custom threshold
outliers_z = utils.detect_outliers(df, column="value", method="zscore", threshold=2.5)
print(df[outliers_z])

# Isolation Forest
outliers_if = utils.detect_outliers(
    df, column="value", method="isolation_forest", contamination=0.1
)
print(df[outliers_if])`})})]}),(0,d.jsxs)("section",{className:"mb-12",children:[(0,d.jsx)("h2",{className:"section-title",children:"Effect Size"}),(0,d.jsx)("div",{className:"method-list",children:(0,d.jsx)(g.MethodCard,{name:"calculate_effect_size",signature:"calculate_effect_size(data: pd.DataFrame | None = None, group1: pd.Series | list | None = None, group2: pd.Series | list | None = None, method: Literal['cohen', 'hedges'] = 'cohen') -> dict",description:"Calculate the standardised effect size between two groups using Cohen's d or Hedges' g. Accepts either a DataFrame with a column to split on or two separate data series.",parameters:[{name:"data",type:"pd.DataFrame | None",description:"Optional DataFrame containing both groups (used with column parameter).",default:"None"},{name:"group1",type:"pd.Series | list | None",description:"First group of values.",default:"None"},{name:"group2",type:"pd.Series | list | None",description:"Second group of values.",default:"None"},{name:"method",type:"'cohen' | 'hedges'",description:"Effect size metric. 'cohen' uses pooled standard deviation; 'hedges' applies a small-sample correction factor.",default:"'cohen'"}],returns:"dict",example:`import pandas as pd
from stats_lib import UtilsStats

utils = UtilsStats()

# Using two separate series
control = [52, 55, 58, 57, 54, 56]
treatment = [65, 68, 72, 70, 66, 71]

result = utils.calculate_effect_size(
    group1=control,
    group2=treatment,
    method="cohen"
)
print(result)
# {'effect_size': 2.14, 'method': 'cohen', 'interpretation': 'large'}

# Hedges' g for small samples
result_g = utils.calculate_effect_size(
    group1=control,
    group2=treatment,
    method="hedges"
)
print(result_g)`})})]}),(0,d.jsxs)("section",{className:"mb-12",children:[(0,d.jsx)("h2",{className:"section-title",children:"Descriptive Statistics"}),(0,d.jsx)("div",{className:"method-list",children:(0,d.jsx)(g.MethodCard,{name:"get_descriptive_stats",signature:"get_descriptive_stats(data: pd.DataFrame, column: str | None = None) -> dict",description:"Compute a comprehensive set of descriptive statistics for a dataset or specific column. Returns a dictionary with count, mean, standard deviation, min, max, quartiles, skewness, and kurtosis.",parameters:[{name:"data",type:"pd.DataFrame",description:"Input DataFrame."},{name:"column",type:"str | None",description:"Column name. If None, computes stats for all numeric columns.",default:"None"}],returns:"dict",example:`import pandas as pd
from stats_lib import UtilsStats

utils = UtilsStats()

df = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

stats = utils.get_descriptive_stats(df, column="A")
print(stats)
# {'count': 10, 'mean': 5.5, 'std': 3.028, 'min': 1.0,
#  'q25': 3.25, 'median': 5.5, 'q75': 7.75, 'max': 10.0,
#  'skewness': 0.0, 'kurtosis': -1.224}

# All numeric columns
all_stats = utils.get_descriptive_stats(df)
print(all_stats)`})})]}),(0,d.jsxs)("section",{className:"mb-12",children:[(0,d.jsx)("h2",{className:"section-title",children:"Plotting Methods"}),(0,d.jsxs)("p",{className:"text-sm text-muted leading-relaxed mb-4",children:["These methods generate visualisations using the backend configured via"," ",(0,d.jsx)("code",{className:"code-inline",children:"set_plot_backend"}),". Each returns a figure object that can be further customised or saved."]}),(0,d.jsxs)("div",{className:"method-list",children:[(0,d.jsx)(g.MethodCard,{name:"plot_distribution",signature:"plot_distribution(data: pd.DataFrame, column: str | None = None, plot_type: str = 'hist', backend: str = 'seaborn', bins: int = 30, figsize: tuple | None = None, save_fig: bool | None = None, filename: str | None = None) -> matplotlib.figure.Figure | plotly.graph_objects.Figure",description:"Plot the distribution of a numeric column using histograms, KDE, box plots, or violin plots.",parameters:[{name:"data",type:"pd.DataFrame",description:"Input DataFrame."},{name:"column",type:"str | None",description:"Column name to plot. If None, plots all numeric columns.",default:"None"},{name:"plot_type",type:"str",description:"Type of plot: 'hist', 'kde', 'box', 'violin'.",default:"'hist'"},{name:"backend",type:"str",description:"Visualisation backend.",default:"'seaborn'"},{name:"bins",type:"int",description:"Number of bins for histograms.",default:"30"},{name:"figsize",type:"tuple | None",description:"Figure size as (width, height). Uses default if None.",default:"None"},{name:"save_fig",type:"bool | None",description:"Override the global save_fig setting.",default:"None"},{name:"filename",type:"str | None",description:"Filename for saving. Auto-generated if None.",default:"None"}],returns:"matplotlib.figure.Figure | plotly.graph_objects.Figure",example:`import pandas as pd
from stats_lib import UtilsStats

utils = UtilsStats()

df = pd.DataFrame({"values": np.random.randn(1000)})

# Histogram
fig = utils.plot_distribution(df, column="values", plot_type="hist", bins=50)

# KDE plot
fig = utils.plot_distribution(df, column="values", plot_type="kde")

# Box plot
fig = utils.plot_distribution(df, column="values", plot_type="box")`}),(0,d.jsx)(g.MethodCard,{name:"plot_correlation_matrix",signature:"plot_correlation_matrix(data: pd.DataFrame, method: str = 'pearson', backend: str = 'seaborn', triangular: bool = False, figsize: tuple | None = None, save_fig: bool | None = None, filename: str | None = None) -> matplotlib.figure.Figure | plotly.graph_objects.Figure",description:"Plot a correlation matrix heatmap for numeric columns using the specified correlation method.",parameters:[{name:"data",type:"pd.DataFrame",description:"Input DataFrame."},{name:"method",type:"str",description:"Correlation method: 'pearson', 'spearman', or 'kendall'.",default:"'pearson'"},{name:"backend",type:"str",description:"Visualisation backend.",default:"'seaborn'"},{name:"triangular",type:"bool",description:"Whether to show only the lower triangle of the matrix.",default:"False"},{name:"figsize",type:"tuple | None",description:"Figure size. Uses default if None.",default:"None"},{name:"save_fig",type:"bool | None",description:"Override the global save_fig setting.",default:"None"},{name:"filename",type:"str | None",description:"Filename for saving.",default:"None"}],returns:"matplotlib.figure.Figure | plotly.graph_objects.Figure",example:`import pandas as pd
from stats_lib import UtilsStats

utils = UtilsStats()

df = pd.DataFrame({
    "A": np.random.randn(100),
    "B": np.random.randn(100),
    "C": np.random.randn(100),
    "D": np.random.randn(100)
})

# Full correlation matrix
fig = utils.plot_correlation_matrix(df, method="pearson")

# Lower triangular heatmap
fig = utils.plot_correlation_matrix(df, method="spearman", triangular=True)`}),(0,d.jsx)(g.MethodCard,{name:"plot_scatter_matrix",signature:"plot_scatter_matrix(data: pd.DataFrame, columns: list[str] | None = None, backend: str = 'seaborn', figsize: tuple | None = None, save_fig: bool | None = None, filename: str | None = None) -> matplotlib.figure.Figure | plotly.graph_objects.Figure",description:"Generate a scatter matrix (pairplot) to visualise pairwise relationships between numeric columns.",parameters:[{name:"data",type:"pd.DataFrame",description:"Input DataFrame."},{name:"columns",type:"list[str] | None",description:"Subset of columns to include. If None, uses all numeric columns.",default:"None"},{name:"backend",type:"str",description:"Visualisation backend.",default:"'seaborn'"},{name:"figsize",type:"tuple | None",description:"Figure size. Uses default if None.",default:"None"},{name:"save_fig",type:"bool | None",description:"Override the global save_fig setting.",default:"None"},{name:"filename",type:"str | None",description:"Filename for saving.",default:"None"}],returns:"matplotlib.figure.Figure | plotly.graph_objects.Figure",example:`import pandas as pd
from stats_lib import UtilsStats

utils = UtilsStats()

df = pd.DataFrame({
    "height": np.random.randn(100) * 10 + 170,
    "weight": np.random.randn(100) * 15 + 70,
    "age": np.random.randint(20, 60, 100)
})

# Full scatter matrix
fig = utils.plot_scatter_matrix(df)

# Selected columns
fig = utils.plot_scatter_matrix(df, columns=["height", "weight"])`}),(0,d.jsx)(g.MethodCard,{name:"plot_distribution_with_ci",signature:"plot_distribution_with_ci(data: pd.DataFrame, column: str | None = None, confidence_level: float = 0.95, ci_method: str = 'parametric', bins: int = 30, figsize: tuple | None = None, save_fig: bool | None = None, filename: str | None = None) -> matplotlib.figure.Figure | plotly.graph_objects.Figure",description:"Plot a distribution histogram with an overlaid confidence interval for the mean. Supports both parametric and bootstrap CI methods.",parameters:[{name:"data",type:"pd.DataFrame",description:"Input DataFrame."},{name:"column",type:"str | None",description:"Column name to plot.",default:"None"},{name:"confidence_level",type:"float",description:"Confidence level for the interval.",default:"0.95"},{name:"ci_method",type:"'parametric' | 'bootstrap'",description:"Method for CI calculation.",default:"'parametric'"},{name:"bins",type:"int",description:"Number of histogram bins.",default:"30"},{name:"figsize",type:"tuple | None",description:"Figure size. Uses default if None.",default:"None"},{name:"save_fig",type:"bool | None",description:"Override the global save_fig setting.",default:"None"},{name:"filename",type:"str | None",description:"Filename for saving.",default:"None"}],returns:"matplotlib.figure.Figure | plotly.graph_objects.Figure",example:`import pandas as pd
import numpy as np
from stats_lib import UtilsStats

utils = UtilsStats()

df = pd.DataFrame({"score": np.random.randn(200) * 15 + 75})

# Distribution with parametric CI
fig = utils.plot_distribution_with_ci(
    df, column="score",
    confidence_level=0.95,
    ci_method="parametric",
    bins=40
)

# Distribution with bootstrap CI
fig = utils.plot_distribution_with_ci(
    df, column="score",
    confidence_level=0.99,
    ci_method="bootstrap"
)`})]})]})]})}},6439:a=>{"use strict";a.exports=require("next/dist/shared/lib/no-fallback-error.external")},6713:a=>{"use strict";a.exports=require("next/dist/shared/lib/router/utils/is-bot")},8354:a=>{"use strict";a.exports=require("util")},9121:a=>{"use strict";a.exports=require("next/dist/server/app-render/action-async-storage.external.js")},9294:a=>{"use strict";a.exports=require("next/dist/server/app-render/work-async-storage.external.js")},9902:a=>{"use strict";a.exports=require("path")}};var b=require("../../../webpack-runtime.js");b.C(a);var c=b.X(0,[424,853,472],()=>b(b.s=1201));module.exports=c})();