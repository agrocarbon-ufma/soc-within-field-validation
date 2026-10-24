"""Analysis code for: "Between-system contrasts can inflate remote sensing accuracy for soil carbon:
a call for within-field validation" (Oliveira et al., submitted to Precision Agriculture).

Fazenda Barbosa, Brejo, Maranhão, Brazil. Systems (internal codes used in the code -> codes in the article):
MN -> NF (native forest, cerrado sensu stricto); ILP8 -> ICL (integrated crop-livestock system, 8 years);
PM -> MP (managed pasture). 20 points per system; layers 0-10, 10-20 and 20-30 cm (0-100, 100-200, 200-300 mm);
dry season (2 July 2024) and wet season (1 April 2025).

Data are read from ../data (samples.csv, points.csv, covariates.csv). Function names, variable names and code
comments are in Portuguese: COT = SOC, COP = POC, COM = MAOC (all in g/kg); DR_ = UAV, PL_ = PlanetScope,
S2_ = Sentinel-2 single date (dry season 2024), S2c_ = Sentinel-2 wet-season bare-soil composite (2025),
S2s_ = Sentinel-2 time series 2019-2024.
"""
import os, re, json, time
import numpy as np, pandas as pd


CAMADAS = ['0-10cm', '10-20cm', '20-30cm']
CORES = {'MN': '#5E3C99', 'ILP8': '#1B7837', 'PM': '#E69F00'}   # validadas para daltonismo (ΔE CVD ≥ 19)
MARCAS = {'MN': 'o', 'ILP8': 's', 'PM': '^'}                       # forma redundante com a cor
NOMES = {'MN': 'Native forest', 'ILP8': 'ICL', 'PM': 'Managed pasture'}
MM = 1 / 25.4   # milímetro em polegadas (Nature: 89 mm uma coluna, 183 mm duas colunas)



# ----------------------------------------------------------------------------- dados (arquivos públicos em ../data)
SIS_PUB = {'NF': 'MN', 'ICL': 'ILP8', 'MP': 'PM'}
COV_PUB = {'UAV_': 'DR_', 'PS_': 'PL_', 'S2wet_': 'S2c_', 'S2ts_': 'S2s_', 'S2_': 'S2_'}
COV_EXTRA = {'TPI_150m': 'TPI150', 'Slope_deg': 'decliv', 'Elevation_m': 'elev', 'SG_clay_g_kg': 'SG_clay', 'SG_sand_g_kg': 'SG_sand'}
VERSOES = {'corrected': ('SOC (g kg⁻¹)', 'POC (g kg⁻¹)'), 'uncorrected': ('SOC uncorrected (g kg⁻¹)', 'POC uncorrected (g kg⁻¹)')}


def _nome_cov(c):
    if c in COV_EXTRA:
        return COV_EXTRA[c]
    for p, q in COV_PUB.items():
        if c.startswith(p):
            return q + c[len(p):]
    return c


def ler_amostras(pasta, versao='corrected'):
    """samples.csv em formato longo: Tratamento, Amostra, Camada, epoca, COT, COP, COM (g/kg).
    versao='corrected' (valores do artigo) ou 'uncorrected' (análise de sensibilidade)."""
    s = pd.read_csv(os.path.join(pasta, 'samples.csv'))
    cot, cop = VERSOES[versao]
    top = s['Depth top (mm)'] // 10; bot = s['Depth bottom (mm)'] // 10
    return pd.DataFrame({'Tratamento': s.System.map(SIS_PUB).values, 'Amostra': s.Point.astype(int).values,
                         'Camada': (top.astype(str) + '-' + bot.astype(str) + 'cm').values,
                         'epoca': s.Season.map({'dry': 'seco', 'wet': 'chuvoso'}).values,
                         'COT': s[cot].values, 'COP': s[cop].values, 'COM': (s[cot] - s[cop]).values})


def carregar(pasta='../data', versao='corrected', z_mad_cop=3.5):
    """Período seco: solo das três camadas + coordenadas + covariáveis. Uma linha por ponto e camada.
    O COP com |z| > 3,5 (MAD, dentro de sistema x camada) fica marcado em flag_COP para a análise de sensibilidade."""
    s = ler_amostras(pasta, versao)
    d = s[s.epoca == 'seco'].drop(columns='epoca')
    p = pd.read_csv(os.path.join(pasta, 'points.csv'))
    p = pd.DataFrame({'Tratamento': p.System.map(SIS_PUB), 'Amostra': p.Point.astype(int),
                      'lon': p['Longitude (WGS 84)'], 'lat': p['Latitude (WGS 84)'],
                      'E': p['E (UTM 23S, m)'], 'N': p['N (UTM 23S, m)']})
    c = pd.read_csv(os.path.join(pasta, 'covariates.csv'))
    c['Tratamento'] = c.Point_ID.str.split('-').str[0].map(SIS_PUB)
    c['Amostra'] = c.Point_ID.str.split('-').str[1].astype(int)
    c = c.drop(columns='Point_ID').rename(columns=_nome_cov)
    d = d.merge(p, on=['Tratamento', 'Amostra'], how='left').merge(c, on=['Tratamento', 'Amostra'], how='left')
    z = d.groupby(['Tratamento', 'Camada'])['COP'].transform(
        lambda x: (x - x.median()) / (1.4826 * (x - x.median()).abs().median()))
    d['flag_COP'] = z.abs() > z_mad_cop
    print(f'{len(d)} amostras do período seco ({versao}); COP marcado (sensibilidade):',
          d.loc[d.flag_COP, ['Tratamento', 'Amostra', 'Camada', 'COP']].values.tolist())
    return d


def tabela_ponto(d):
    """Uma linha por ponto: COT, COP e COM por camada (ex.: COP_0-10) e as covariáveis."""
    w = d.pivot_table(index=['Tratamento', 'Amostra'], columns='Camada', values=['COP', 'COT', 'COM'], dropna=False)
    w.columns = [f'{v}_{c.replace("cm", "")}' for v, c in w.columns]
    cov = d.drop(columns=['Camada', 'COP', 'COT', 'COM', 'flag_COP'], errors='ignore').drop_duplicates(['Tratamento', 'Amostra'])
    # marca de sensibilidade do COP: só a de 0-10 cm (pontos 6-9 do ILP8), a camada dos modelos principais
    top = d[d.Camada == '0-10cm'] if (d.Camada == '0-10cm').any() else d
    fl = top.groupby(['Tratamento', 'Amostra'])['flag_COP'].any().rename('flag_COP')
    return w.reset_index().merge(cov, on=['Tratamento', 'Amostra']).merge(fl.reset_index(), on=['Tratamento', 'Amostra'])


def adicionar_chuvoso(w, pasta='../data', versao='corrected'):
    """COT, COP e COM de 0-10 cm do período chuvoso, nos mesmos pontos (sufixo _ch)."""
    s = ler_amostras(pasta, versao)
    c = s[(s.epoca == 'chuvoso') & (s.Camada == '0-10cm')][['Tratamento', 'Amostra', 'COT', 'COP', 'COM']].rename(
        columns={'COT': 'COT_0-10_ch', 'COP': 'COP_0-10_ch', 'COM': 'COM_0-10_ch'})
    w = w.drop(columns=[k for k in ['COT_0-10_ch', 'COP_0-10_ch', 'COM_0-10_ch'] if k in w.columns])
    return w.merge(c, on=['Tratamento', 'Amostra'], how='left')


def estilo_publicacao():
    """Padrão de figura para Nature/Science: sem serifa 7 pt, eixos finos, sem bordas superior e direita,
    texto editável no PDF."""
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
        'legend.fontsize': 6, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.direction': 'out', 'ytick.direction': 'out',
        'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
        'savefig.dpi': 600, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})


def salvar(fig, pasta, nome):
    """PNG 600 dpi para conferir e PDF vetorial para a revista."""
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(pasta, f'{nome}.{ext}'))


def metricas(y, yp, media_sis=None):
    y, yp = np.asarray(y, float), np.asarray(yp, float)
    sse = ((y - yp) ** 2).sum()
    m = dict(R2=1 - sse / ((y - y.mean()) ** 2).sum(), RMSE=np.sqrt(sse / len(y)),
             MAPE=100 * np.mean(np.abs((y - yp) / y)), vies=np.mean(yp - y))
    if media_sis is not None:
        m['R2_dentro'] = 1 - sse / ((y - media_sis) ** 2).sum()
    return m


def ic_r2(y, yp, n_boot=2000, seed=1):
    """IC 95% do R² das predições (bootstrap dos pontos)."""
    rng = np.random.default_rng(seed); y, yp = np.asarray(y), np.asarray(yp); v = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if np.ptp(y[i]) > 0:
            v.append(1 - ((y[i] - yp[i]) ** 2).sum() / ((y[i] - y[i].mean()) ** 2).sum())
    return np.percentile(v, [2.5, 97.5])
# Índices do drone e do Planet com fórmula correta na Tabela 2 da dissertação.
# EVI e SAVI (fórmulas erradas), REIP e CO2Flux (bandas que os sensores não têm) ficam de fora.
IDX_OK = ['NDVI', 'NDRE', 'NDWI']
N_ARVORES = 300


# ----------------------------------------------------------------------------- Earth Engine
def coletar_covariaveis(cfg, d):
    """Sentinel-2 (data da coleta e série), MDE Copernicus 30 m e textura, nos 60 pontos.
    Uma requisição por grupo de variáveis, com nova tentativa em caso de erro 429."""
    import ee, ee.oauth
    # ~/.config não é gravável neste Mac: a chave fica em ~/earthengine_chave (a mesma dos outros notebooks)
    chave = os.path.expanduser(cfg.get('PASTA_CHAVE_GEE', '~/earthengine_chave'))
    os.makedirs(chave, exist_ok=True)
    ee.oauth.get_credentials_path = lambda: os.path.join(chave, 'credentials')
    try:
        ee.Initialize(project=cfg['PROJETO_GEE'])
    except Exception:
        ee.Authenticate(auth_mode='notebook', force=True)
        ee.Initialize(project=cfg['PROJETO_GEE'])
    pts = d.drop_duplicates(['Tratamento', 'Amostra'])[['Tratamento', 'Amostra', 'lon', 'lat']].dropna()
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([r.lon, r.lat]).buffer(cfg.get('BUFFER_M', 10)),
                                          {'Tratamento': r.Tratamento, 'Amostra': int(r.Amostra)})
                               for r in pts.itertuples()])
    regiao = fc.geometry().bounds().buffer(500)

    def mascara(img):
        scl = img.select('SCL')
        ok = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
        return img.updateMask(ok).divide(10000).copyProperties(img, ['system:time_start'])

    def indices(img):
        img = ee.Image(img)
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        ndre = img.normalizedDifference(['B8A', 'B5']).rename('NDRE')
        ndwi = img.normalizedDifference(['B3', 'B8']).rename('NDWI')
        ndmi = img.normalizedDifference(['B8', 'B11']).rename('NDMI')
        evi = img.expression('2.5*(N-R)/(N+6*R-7.5*B+1)', {'N': img.select('B8'), 'R': img.select('B4'), 'B': img.select('B2')}).rename('EVI')
        savi = img.expression('1.5*(N-R)/(N+R+0.5)', {'N': img.select('B8'), 'R': img.select('B4')}).rename('SAVI')
        bsi = img.expression('((S+R)-(N+B))/((S+R)+(N+B))', {'S': img.select('B11'), 'R': img.select('B4'), 'N': img.select('B8'), 'B': img.select('B2')}).rename('BSI')
        return img.addBands([ndvi, ndre, ndwi, ndmi, evi, savi, bsi])

    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(regiao)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40)).map(mascara))
    bandas = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    ini, fim = cfg['S2_JANELA']
    cena = indices(s2.filterDate(ini, fim).select(bandas).median()).select(bandas + ['NDVI', 'NDRE', 'NDWI', 'NDMI', 'EVI', 'SAVI', 'BSI'])
    cena = cena.rename([f'S2_{b}' for b in cena.bandNames().getInfo()])

    # cena da época chuvosa (solo coletado em abril/2025), prefixo S2c_, para a validação no tempo.
    # Em abril o ILP8 está com soja: a janela fica depois da colheita, e em cada pixel entra a data
    # com mais solo exposto (maior BSI), em vez da mediana, que misturaria soja e palhada.
    ci, cf = cfg.get('S2_JANELA_CHUVOSO', ('2025-05-15', '2025-08-31'))
    cena_c = (s2.filterDate(ci, cf).select(bandas).map(indices).qualityMosaic('BSI')
              .select(bandas + ['NDVI', 'NDRE', 'NDWI', 'NDMI', 'EVI', 'SAVI', 'BSI']))
    cena_c = cena_c.rename([f'S2c_{b}' for b in cena_c.bandNames().getInfo()])

    a0, a1 = cfg['S2_SERIE_ANOS']
    serie = s2.filterDate(f'{a0}-01-01', f'{a1}-12-31').select(bandas).map(indices)
    seca = serie.filter(ee.Filter.calendarRange(6, 10, 'month'))
    chuva = serie.filter(ee.Filter.calendarRange(1, 5, 'month'))
    exposto = serie.map(lambda i: i.select('NDVI').lt(0.25).And(i.select('BSI').gt(0)).rename('exp'))
    hist = ee.Image.cat([
        seca.select('NDVI').mean().rename('S2s_NDVI_seca'),
        chuva.select('NDVI').mean().rename('S2s_NDVI_chuva'),
        serie.select('NDVI').reduce(ee.Reducer.stdDev()).rename('S2s_NDVI_dp'),
        seca.select('BSI').mean().rename('S2s_BSI_seca'),
        exposto.mean().rename('S2s_freq_solo_exposto'),
    ])

    dem = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM').filterBounds(regiao).mosaic().setDefaultProjection('EPSG:4326', None, 30)
    terreno = ee.Terrain.products(dem)
    tpi = dem.subtract(dem.focalMean(150, 'circle', 'meters')).rename('TPI150')
    relevo = ee.Image.cat([dem.rename('elev'), terreno.select('slope').rename('decliv'), tpi])

    sg = ee.Image.cat([ee.Image(f'projects/soilgrids-isric/{v}_mean').select([f'{v}_0-5cm_mean', f'{v}_5-15cm_mean', f'{v}_15-30cm_mean'])
                       .reduce(ee.Reducer.mean()).divide(10).rename(f'SG_{v}') for v in ['clay', 'sand']])
    camadas = [('S2', cena, 10), ('S2c', cena_c, 10), ('S2s', hist, 10), ('relevo', relevo, 30), ('textura', sg, 250)]
    mb = cfg.get('MAPBIOMAS_TEXTURA', {})
    for nome, asset in mb.items():
        try:
            img = ee.Image(asset); img.bandNames().getInfo()
            camadas.append((f'MB_{nome}', img.rename(img.bandNames().map(lambda b: ee.String(f'MB_{nome}_').cat(b))), 30))
        except Exception as e:
            print(f'MapBiomas {nome} não abriu ({e}); seguindo sem ele.')

    saida = os.path.join(cfg['PASTA'], cfg.get('COVARIAVEIS_CSV', 'covariaveis_gee.csv'))
    tabs = []
    fc_pt = fc.map(lambda f: f.setGeometry(f.geometry().centroid(1)))
    for nome, img, esc in camadas:
        # pixel maior que o buffer (relevo 30 m, textura 250 m): valor do pixel no ponto, senão a média sai vazia
        alvo = fc if esc <= 10 else fc_pt
        red = ee.Reducer.mean() if esc <= 10 else ee.Reducer.first()
        for t in range(5):
            try:
                r = img.reduceRegions(alvo, red, esc).getInfo()
                tabs.append(pd.DataFrame([f['properties'] for f in r['features']]).set_index(['Tratamento', 'Amostra']))
                print('ok:', nome); break
            except Exception as e:
                if '429' in str(e) or 'Too many' in str(e):
                    time.sleep(2 ** t * 5); continue
                raise
    out = pd.concat(tabs, axis=1).reset_index()
    out.to_csv(saida, index=False)
    print('salvo', saida, out.shape)
    return out


# ----------------------------------------------------------------------------- estatística
def comparar_sistemas(d):
    """Modelo misto: variável ~ sistema * camada, ponto como efeito aleatório."""
    import statsmodels.formula.api as smf
    from scipy import stats
    res = {}
    for v in [c for c in ['COT', 'COP', 'COM'] if c in d.columns and d[c].notna().any()]:   # 22/10/2026: teor e frações (sem estoque)
        x = d[['Tratamento', 'Amostra', 'Camada', v]].dropna().copy()
        x['pt'] = x.Tratamento + '_' + x.Amostra.astype(str)
        x['y'] = np.log(x[v])
        # com alguma combinação sistema x camada vazia (ex.: PM 20-30 removida), o modelo fatorial fica singular;
        # nesse caso usa uma média por combinação (mesmo ajuste, sem o termo que não pode ser estimado)
        completo = x.groupby(['Tratamento', 'Camada']).size().reindex(
            pd.MultiIndex.from_product([x.Tratamento.unique(), x.Camada.unique()]), fill_value=0).min() > 0
        x['cel'] = x.Tratamento + ' ' + x.Camada
        if x.Camada.nunique() == 1:
            # uma camada só (topsoil): uma observação por ponto, então basta o modelo linear por sistema
            m = smf.ols('y ~ C(Tratamento)', x).fit()
        else:
            formula = 'y ~ C(Tratamento) * C(Camada)' if completo else 'y ~ C(cel)'
            m = smf.mixedlm(formula, x, groups=x['pt']).fit(reml=True)
        sw = stats.shapiro(m.resid)
        desc = x.groupby(['Camada', 'Tratamento'])[v].agg(['mean', 'sem', 'count']).round(3)
        # contraste entre sistemas dentro de cada camada (Kruskal-Wallis, robusto)
        kw = {}
        for c in [c for c in CAMADAS if c in set(x.Camada)]:
            gs = [g[v].values for _, g in x[x.Camada == c].groupby('Tratamento')]
            kw[c] = stats.kruskal(*gs).pvalue if len(gs) > 1 else np.nan
        # pares de sistemas (Mann-Whitney, correção de Holm)
        from itertools import combinations
        pares = []
        for c in sorted(set(x.Camada)):
            g = {t: z[v].values for t, z in x[x.Camada == c].groupby('Tratamento')}
            for a, b in combinations(sorted(g), 2):
                pares.append(dict(Camada=c, par=f'{a} x {b}', p=stats.mannwhitneyu(g[a], g[b]).pvalue))
        pares = pd.DataFrame(pares)
        if len(pares):
            # Holm dentro de cada camada (três pares de sistemas por camada)
            def holm(pv):
                o = np.argsort(pv); k = len(o); adj = np.empty(k); acc = 0
                for i, j in enumerate(o):
                    acc = max(acc, min(1, (k - i) * pv[j])); adj[j] = acc
                return adj
            pares['p_Holm'] = np.concatenate([holm(g.p.values) for _, g in pares.groupby('Camada', sort=False)])
        res[v] = dict(modelo=m, shapiro_p=sw.pvalue, descr=desc, kw=kw, pares=pares)
        tab = m.summary().tables[1]
        print(f'\n=== {v} (log) ===\n', tab.round(4) if hasattr(tab, 'round') else tab)
        print('Shapiro dos resíduos p =', round(sw.pvalue, 4), '| KW por camada:', {k: round(p, 4) for k, p in kw.items()})
    return res


def correlacao_decomposta(w, alvos, covs):
    """r com todos os pontos (entre + dentro) e r dentro dos sistemas (centrado na média de cada sistema)."""
    from scipy import stats
    lin = []
    for a in alvos:
        for c in covs:
            x = w[['Tratamento', a, c]].dropna()
            if len(x) < 10 or x[c].std() == 0:
                continue
            r_tot, p_tot = stats.pearsonr(x[c], x[a])
            xc = x[[a, c]] - x.groupby('Tratamento')[[a, c]].transform('mean')
            r_den, p_den = stats.pearsonr(xc[c], xc[a])
            lin.append(dict(alvo=a, cov=c, r_total=r_tot, p_total=p_tot, r_dentro=r_den, p_dentro=p_den, n=len(x)))
    t = pd.DataFrame(lin)
    if len(t):
        from statsmodels.stats.multitest import multipletests
        t['p_dentro_FDR'] = multipletests(t.p_dentro, method='fdr_bh')[1]
        t['p_total_FDR'] = multipletests(t.p_total, method='fdr_bh')[1]
    return t


def _modelo(tipo, p=3):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.linear_model import LinearRegression
    if tipo == 'RF':
        return RandomForestRegressor(n_estimators=N_ARVORES, min_samples_leaf=3, random_state=42, n_jobs=-1)
    if tipo == 'PLSR':
        return GridSearchCV(make_pipeline(StandardScaler(), PLSRegression()),
                            {'plsregression__n_components': list(range(1, min(3, p) + 1))}, cv=KFold(5, shuffle=True, random_state=1),
                            scoring='neg_mean_squared_error')
    if tipo == 'OLS':
        return LinearRegression()


def avaliar(w, alvo, conjuntos, modelos=('RF', 'PLSR'), excluir_flag=False):
    """LOOCV aninhada (escala e nº de componentes dentro de cada dobra).
    R2_cv: com todos os pontos. R2_dentro: quanto o modelo explica além da média de cada sistema
    (> 0 significa que bate o modelo que só sabe o sistema)."""
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    x = w.copy()
    if excluir_flag:
        x = x[~x.flag_COP]
    lin, pred = [], {}
    for nome, cols in conjuntos.items():
        cols = [c for c in cols if c in x.columns or c.startswith('sis_')]
        # covariável com valor faltante em mais de 5% dos pontos sai do conjunto, em vez de derrubar os pontos
        falt = [c for c in cols if not c.startswith('sis_') and x[c].isna().mean() > 0.05]
        if falt:
            print(f'{alvo} | {nome}: sem {falt} (faltantes)')
            cols = [c for c in cols if c not in falt]
        sis = [c for c in cols if c.startswith('sis_')]
        num = [c for c in cols if c not in sis]
        z = x.dropna(subset=[alvo] + num)
        dum = pd.get_dummies(z.Tratamento, prefix='sis', dtype=float)
        sis = [c for c in sis if c in dum.columns and dum[c].nunique() > 1]   # só sistemas presentes neste subconjunto
        if len(sis) == 2 and dum.shape[1] == 2:
            sis = sis[:1]   # com dois sistemas, uma coluna basta (a outra é redundante)
        if len(z) < 15 or not (num or sis):
            continue
        X = pd.concat([z[num], dum[sis] if sis else None], axis=1).values
        y = z[alvo].values
        media_sis = z.groupby('Tratamento')[alvo].transform('mean').values
        for mt in modelos:
            if mt == 'PLSR' and X.shape[1] < 2:
                mt_ = 'OLS'
            else:
                mt_ = mt
            yp = cross_val_predict(_modelo(mt_, X.shape[1]), X, y, cv=LeaveOneOut()).ravel()
            sse = ((y - yp) ** 2).sum()
            mt = metricas(y, yp, media_sis); lo, hi = ic_r2(y, yp)
            lin.append(dict(alvo=alvo, conjunto=nome, modelo=mt_, n=len(y), p=X.shape[1],
                            R2_cv=mt['R2'], R2_IC95_inf=lo, R2_IC95_sup=hi, R2_dentro=mt['R2_dentro'],
                            RMSE_cv=mt['RMSE'], MAPE_cv=mt['MAPE'], vies=mt['vies']))
            pred[(nome, mt_)] = pd.DataFrame({'Tratamento': z.Tratamento.values, 'obs': y, 'pred': yp})
    return pd.DataFrame(lin), pred


def validar_no_tempo(w, alvo_seco, alvo_chuv, conjuntos, modelos=('RF', 'PLSR'), excluir_flag=False):
    """Calibra com todos os pontos do seco e prevê o chuvoso nos mesmos pontos.
    A cena Sentinel-2 da data (S2_) é trocada pela cena do chuvoso (S2c_); série, relevo e textura não mudam.
    Drone e Planet só existem no seco, então conjuntos com eles ficam de fora.
    Além do R² com todos os pontos, mede a transferência em dois níveis:
      - dentro dos sistemas: r_dentro_val (obs e pred centrados na média de cada sistema; só a ordem dos pontos)
        e R2_dentro_val (erro comparado com a média observada de cada sistema no chuvoso);
      - nível do sistema: erro (%) da média predita de cada sistema e se a ordem dos sistemas se mantém.
    excluir_flag=True tira da calibração o COP marcado (pontos 6-9 do ILP8).
    Devolve a tabela de métricas e as predições (para o gráfico observado x predito)."""
    lin, pred = [], {}
    if excluir_flag and 'flag_COP' in w.columns:
        w = w[~w.flag_COP.fillna(False).astype(bool)]
    for nome, cols in conjuntos.items():
        if any(c.startswith(('DR_', 'PL_')) for c in cols):
            continue
        sis = [c for c in cols if c.startswith('sis_')]
        num = [c for c in cols if c in w.columns and not c.startswith('sis_')]
        num_ch = [c.replace('S2_', 'S2c_', 1) if c.startswith('S2_') else c for c in num]
        if any(c not in w.columns for c in num_ch):
            print(f'{nome}: falta a cena do chuvoso (S2c_); apague covariaveis_gee.csv e colete de novo.'); continue
        tr = w.dropna(subset=[alvo_seco] + num); te = w.dropna(subset=[alvo_chuv] + num_ch)
        te = te[te.Tratamento.isin(tr.Tratamento.unique())]   # só valida sistemas que existem na calibração
        d_tr = pd.get_dummies(tr.Tratamento, prefix='sis', dtype=float)
        sis = [c for c in sis if c in d_tr.columns and d_tr[c].nunique() > 1][:max(1, tr.Tratamento.nunique() - 1)]
        dum = lambda z: pd.get_dummies(z.Tratamento, prefix='sis', dtype=float).reindex(columns=sis, fill_value=0.0)
        Xtr = pd.concat([tr[num], dum(tr)], axis=1).values
        Xte = pd.concat([te[num_ch].set_axis(num, axis=1), dum(te)], axis=1).values
        if Xtr.shape[1] == 0 or len(te) < 10:
            continue
        y = te[alvo_chuv].values
        g_te = te.Tratamento.values
        media_sis = pd.Series(y).groupby(g_te).transform('mean').values
        centra = lambda v: v - pd.Series(v).groupby(g_te).transform('mean').values
        for mt in modelos:
            mt_ = 'OLS' if mt == 'PLSR' and Xtr.shape[1] < 2 else mt
            yp = _modelo(mt_, Xtr.shape[1]).fit(Xtr, tr[alvo_seco].values).predict(Xte).ravel()
            mt = metricas(y, yp, media_sis); lo, hi = ic_r2(y, yp)
            yc, pc = centra(y), centra(yp)
            r_den = np.corrcoef(yc, pc)[0, 1] if np.std(pc) > 1e-12 else np.nan   # 'Só o sistema' não ordena pontos
            ms = pd.DataFrame({'sis': g_te, 'obs': y, 'pred': yp}).groupby('sis').mean()
            lin.append(dict(alvo=f'{alvo_seco} -> chuvoso', conjunto=nome, modelo=mt_, n_cal=len(tr), n_val=len(te),
                            R2_val=mt['R2'], R2_IC95_inf=lo, R2_IC95_sup=hi, r_val=np.corrcoef(y, yp)[0, 1],
                            R2_dentro_val=mt['R2_dentro'], r_dentro_val=r_den,
                            RMSE_val=mt['RMSE'], MAPE_val=mt['MAPE'], vies=mt['vies'],
                            ordem_sistemas_ok=list(ms.obs.sort_values().index) == list(ms.pred.sort_values().index),
                            **{f'erro_media_{t}_%': 100 * (ms.pred[t] - ms.obs[t]) / ms.obs[t] for t in ms.index}))
            pred[(nome, mt_)] = pd.DataFrame({'Tratamento': g_te, 'obs': y, 'pred': yp})
    return pd.DataFrame(lin), pred


def repetibilidade(w, vs=('COT', 'COP', 'COM'), camada='0-10'):
    """Seco × chuvoso nos mesmos pontos (precisa de adicionar_chuvoso).
    Por sistema: médias, variação (%), testes pareados (t e Wilcoxon) e correlação ponto a ponto
    (Pearson e Spearman). Na linha 'Dentro dos sistemas', as duas épocas são centradas na média de cada
    sistema: mede se a ordem dos pontos dentro do sistema se repete de uma época para a outra.
    Se essa correlação é baixa, a variável não se repete dentro do sistema e nenhum sensor consegue prevê-la ali."""
    from scipy import stats
    lin = []
    for v in vs:
        a, b = f'{v}_{camada}', f'{v}_{camada}_ch'
        if a not in w.columns or b not in w.columns:
            continue
        x = w[['Tratamento', a, b]].dropna()
        for t, q in x.groupby('Tratamento'):
            s_, c_ = q[a].values, q[b].values
            r, pr = stats.pearsonr(s_, c_)
            lin.append(dict(variavel=v, sistema=t, n=len(q), seco=s_.mean(), chuvoso=c_.mean(),
                            variacao_pct=100 * (c_.mean() - s_.mean()) / s_.mean(),
                            p_t=stats.ttest_rel(c_, s_).pvalue, p_wilcoxon=stats.wilcoxon(c_, s_).pvalue,
                            r=r, p_r=pr, rho=stats.spearmanr(s_, c_)[0]))
        s_ = x[a] - x.groupby('Tratamento')[a].transform('mean')
        c_ = x[b] - x.groupby('Tratamento')[b].transform('mean')
        r, pr = stats.pearsonr(s_, c_)
        lin.append(dict(variavel=v, sistema='Dentro dos sistemas', n=len(x), r=r, p_r=pr, rho=stats.spearmanr(s_, c_)[0]))
    return pd.DataFrame(lin)


def conjuntos_padrao(w):
    cs = lambda pref: [c for c in w.columns if c.startswith(pref)]
    s2 = cs('S2_'); s2s = cs('S2s_'); rel = ['elev', 'decliv', 'TPI150']; tex = cs('SG_') + cs('MB_')
    dr = cs('DR_'); pl = cs('PL_'); sis = ['sis_ILP8', 'sis_MN', 'sis_PM']
    tudo = dr + pl + s2 + s2s + rel + tex
    return {
        'Só o sistema': sis,
        'Drone': dr,
        'Planet': pl,
        'Sentinel-2 (data)': s2,
        'Sentinel-2 (série)': s2s,
        'Relevo + textura': rel + tex,
        'Sentinel-2 + relevo + textura': s2 + s2s + rel + tex,
        'Sistema + Sentinel-2 + relevo + textura': sis + s2 + s2s + rel + tex,
        'Todas as covariáveis': tudo,
        'Sistema + todas': sis + tudo,
    }


# ----------------------------------------------------------------------------- mapa e SHAP
SERIE = ['S2s_NDVI_seca', 'S2s_NDVI_chuva', 'S2s_NDVI_dp', 'S2s_BSI_seca', 'S2s_freq_solo_exposto']


def _rf_serie(w, alvo='COP_0-10'):
    """Random Forest com a série Sentinel-2 (o conjunto com sinal dentro dos sistemas), em todos os pontos."""
    from sklearn.ensemble import RandomForestRegressor
    x = w.dropna(subset=[alvo] + SERIE)
    rf = RandomForestRegressor(n_estimators=N_ARVORES, min_samples_leaf=3, random_state=42, n_jobs=-1)
    return rf.fit(x[SERIE].values, x[alvo].values), x


def grade_serie_s2(cfg, w, res_m=10, borda_m=150, saida='grade_serie_s2.csv'):
    """Baixa do Earth Engine as 5 variáveis da série Sentinel-2 em grade de 10 m (UTM 23S) cobrindo as três áreas.
    Mesmas fórmulas de coletar_covariaveis. Salva um csv (E, N e as variáveis) para não baixar de novo."""
    arq = os.path.join(cfg['PASTA'], saida)
    if os.path.exists(arq):
        return pd.read_csv(arq)   # já baixada: não precisa do Earth Engine
    import ee, ee.oauth
    chave = os.path.expanduser(cfg.get('PASTA_CHAVE_GEE', '~/earthengine_chave'))
    ee.oauth.get_credentials_path = lambda: os.path.join(chave, 'credentials')
    try:
        ee.Initialize(project=cfg['PROJETO_GEE'])
    except Exception:
        ee.Authenticate(auth_mode='notebook', force=True); ee.Initialize(project=cfg['PROJETO_GEE'])
    e0, e1 = w.E.min() - borda_m, w.E.max() + borda_m; n0, n1 = w.N.min() - borda_m, w.N.max() + borda_m
    ret = ee.Geometry.Rectangle([e0, n0, e1, n1], ee.Projection('EPSG:31983'), False)

    def mascara(img):
        scl = img.select('SCL')
        ok = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
        return img.updateMask(ok).divide(10000).copyProperties(img, ['system:time_start'])

    def indices(img):
        img = ee.Image(img)
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        bsi = img.expression('((S+R)-(N+B))/((S+R)+(N+B))', {'S': img.select('B11'), 'R': img.select('B4'), 'N': img.select('B8'), 'B': img.select('B2')}).rename('BSI')
        return img.addBands([ndvi, bsi])

    a0, a1 = cfg['S2_SERIE_ANOS']
    serie = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(ret.transform('EPSG:4326', 1))
             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40)).map(mascara)
             .filterDate(f'{a0}-01-01', f'{a1}-12-31').select(['B2', 'B4', 'B8', 'B11']).map(indices))
    seca = serie.filter(ee.Filter.calendarRange(6, 10, 'month')); chuva = serie.filter(ee.Filter.calendarRange(1, 5, 'month'))
    exposto = serie.map(lambda i: i.select('NDVI').lt(0.25).And(i.select('BSI').gt(0)).rename('exp'))
    hist = ee.Image.cat([seca.select('NDVI').mean().rename('S2s_NDVI_seca'), chuva.select('NDVI').mean().rename('S2s_NDVI_chuva'),
                         serie.select('NDVI').reduce(ee.Reducer.stdDev()).rename('S2s_NDVI_dp'),
                         seca.select('BSI').mean().rename('S2s_BSI_seca'), exposto.mean().rename('S2s_freq_solo_exposto')])
    img = hist.reproject(ee.Projection('EPSG:31983').atScale(res_m)).unmask(-9999)
    img = img.addBands(ee.Image.pixelCoordinates(ee.Projection('EPSG:31983').atScale(res_m)).multiply(res_m).add(res_m / 2))
    arr = img.sampleRectangle(region=ret, defaultValue=-9999).getInfo()['properties']
    g = pd.DataFrame({k: np.array(v, float).ravel() for k, v in arr.items()}).rename(columns={'x': 'E', 'y': 'N'})
    g = g.replace(-9999, np.nan)
    g.to_csv(arq, index=False); print(f'Grade salva em {saida}: {len(g)} pixels de {res_m} m')
    return g


def mapa_predito(w, g, alvo='COP_0-10', borda_m=60):
    """Prevê o alvo pixel a pixel com o RF da série e calcula a incerteza (desvio entre as árvores).
    Só dentro das áreas amostradas (envoltória convexa dos pontos de cada sistema + borda) e marca
    os pixels em que alguma variável sai da faixa dos pontos de calibração (extrapolação)."""
    from matplotlib.path import Path
    from scipy.spatial import ConvexHull
    rf, x = _rf_serie(w, alvo)
    g = g.dropna(subset=SERIE).copy(); g['Tratamento'] = None
    for t, q in w.groupby('Tratamento'):
        pts = q[['E', 'N']].dropna().values; h = pts[ConvexHull(pts).vertices]
        c = h.mean(0); h = c + (h - c) * (1 + borda_m / np.linalg.norm(h - c, axis=1).mean())
        g.loc[Path(h).contains_points(g[['E', 'N']].values), 'Tratamento'] = t
    g = g[g.Tratamento.notna()].copy()
    arv = np.stack([a.predict(g[SERIE].values) for a in rf.estimators_])
    g['pred'], g['incerteza'] = arv.mean(0), arv.std(0)
    lo, hi = x[SERIE].min(), x[SERIE].max()
    g['extrapola'] = ((g[SERIE] < lo) | (g[SERIE] > hi)).any(axis=1)
    print(f'{len(g)} pixels nas áreas; {100*g.extrapola.mean():.1f}% fora da faixa de calibração')
    return g


def shap_serie(w, alvo='COP_0-10'):
    """Valores SHAP do RF da série Sentinel-2 (TreeExplainer)."""
    import shap
    rf, x = _rf_serie(w, alvo)
    sv = shap.TreeExplainer(rf).shap_values(x[SERIE].values)
    return dict(shap=pd.DataFrame(sv, columns=SERIE), X=x[SERIE].reset_index(drop=True), sis=x.Tratamento.values, alvo=alvo)


# ----------------------------------------------------------------------------- multivariada
def pca_sistemas(w, solo=('COP_0-10', 'COM_0-10'), covs=None, n_perm=9999, seed=1):
    """PCA (variáveis padronizadas) com carbono do topsoil e índices espectrais, mais PERMANOVA
    (distância euclidiana nas variáveis padronizadas) para testar se os sistemas se separam.
    22/10/2026: as duas frações (COP e COM); o COT fica de fora porque é a soma delas."""
    from sklearn.decomposition import PCA
    covs = covs or [c for c in w.columns if c.startswith(('DR_', 'PL_')) or c in
                    ['S2s_NDVI_seca', 'S2s_NDVI_chuva', 'S2s_BSI_seca', 'S2s_freq_solo_exposto']]
    cols = [c for c in list(solo) + covs if c in w.columns]
    x = w.dropna(subset=cols).reset_index(drop=True)
    Z = ((x[cols] - x[cols].mean()) / x[cols].std(ddof=1)).values
    pca = PCA().fit(Z); sc = pca.transform(Z)
    load = pd.DataFrame(pca.components_[:2].T * np.sqrt(pca.explained_variance_[:2]), index=cols, columns=['PC1', 'PC2'])
    # PERMANOVA: pseudo-F com permutação dos rótulos de sistema
    D2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1); g = x.Tratamento.values; n, k = len(g), len(set(g))
    def F(lab):
        sst = D2[np.triu_indices(n, 1)].sum() / n
        ssw = sum(D2[np.ix_(lab == t, lab == t)][np.triu_indices((lab == t).sum(), 1)].sum() / (lab == t).sum() for t in set(lab))
        return ((sst - ssw) / (k - 1)) / (ssw / (n - k)), 1 - ssw / sst
    f0, r2 = F(g); rng = np.random.default_rng(seed)
    p = (1 + sum(F(rng.permutation(g))[0] >= f0 for _ in range(n_perm))) / (n_perm + 1)
    print(f'PCA com {len(cols)} variáveis e {n} pontos | PC1 {100*pca.explained_variance_ratio_[0]:.1f}%, '
          f'PC2 {100*pca.explained_variance_ratio_[1]:.1f}% | PERMANOVA: pseudo-F = {f0:.1f}, R² = {r2:.2f}, p = {p:.4f}')
    return dict(scores=x[['Tratamento', 'Amostra']].assign(PC1=sc[:, 0], PC2=sc[:, 1]), loadings=load,
                var=pca.explained_variance_ratio_, permanova=dict(F=f0, R2=r2, p=p, n=n))


# ----------------------------------------------------------------------------- figuras
CONJ_EN = {'Só o sistema': 'System only', 'Drone': 'UAV', 'Planet': 'PlanetScope', 'Sentinel-2 (data)': 'Sentinel-2 (single date)',
           'Sentinel-2 (série)': 'Sentinel-2 (time series)', 'Relevo + textura': 'Terrain + texture',
           'Sentinel-2 + relevo + textura': 'Sentinel-2 + terrain + texture',
           'Sistema + Sentinel-2 + relevo + textura': 'System + Sentinel-2 + terrain + texture',
           'Todas as covariáveis': 'All covariates', 'Sistema + todas': 'System + all covariates'}


def rotulo_conj(conj, modelo=None, largura=None):
    import textwrap
    t = CONJ_EN.get(conj, conj) + (f' ({modelo})' if modelo else '')
    return '\n'.join(textwrap.wrap(t, largura)) if largura else t


ROTULOS = {'COP': 'POC (g kg$^{-1}$)', 'COT': 'SOC (g kg$^{-1}$)', 'COM': 'MAOC (g kg$^{-1}$)', 'EST': 'SOC stock (Mg ha$^{-1}$)'}


def fig_obs_pred(pred, alvo, pasta, ncol=4, nome=None):
    """Observado x predito, um painel por conjunto e modelo: pontos por sistema (cor e forma),
    linha 1:1 tracejada, reta ajustada e R², RMSE e MAPE no painel. Largura de duas colunas (183 mm)."""
    import matplotlib.pyplot as plt
    estilo_publicacao()
    chaves = list(pred)
    nl = int(np.ceil(len(chaves) / ncol))
    fig, axs = plt.subplots(nl, ncol, figsize=(183 * MM, 183 * MM / ncol * nl * 1.02), squeeze=False)
    todos = pd.concat(pred.values())
    lo, hi = np.nanpercentile(np.r_[todos.obs, todos.pred], [0, 100]); pad = 0.06 * (hi - lo); lim = (lo - pad, hi + pad)
    var = alvo.split('_')[0]; unid = ROTULOS.get(var, alvo)
    for k, (ax, chave) in enumerate(zip(axs.flat, chaves)):
        p = pred[chave]; mt = metricas(p.obs, p.pred)
        ax.plot(lim, lim, ls=(0, (3, 2)), lw=0.6, color='0.55', zorder=1)
        b1, b0 = np.polyfit(p.obs, p.pred, 1); xx = np.array(lim)
        ax.plot(xx, b0 + b1 * xx, lw=0.8, color='0.15', zorder=2)
        for t in [t for t in ['MN', 'ILP8', 'PM'] if t in set(p.Tratamento)]:
            q = p[p.Tratamento == t]
            ax.scatter(q.obs, q.pred, s=12, marker=MARCAS[t], color=CORES[t], edgecolor='white', linewidth=0.4,
                       label=NOMES[t], zorder=3)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
        ax.text(0.04, 0.96, f'$R^2$ = {mt["R2"]:.2f}\nRMSE = {mt["RMSE"]:.2f}\nMAPE = {mt["MAPE"]:.1f}%',
                transform=ax.transAxes, va='top', ha='left', fontsize=6, linespacing=1.3)
        ax.set_title(rotulo_conj(chave[0], chave[1], 30), loc='left', pad=3, fontsize=6.5)
        ax.text(-0.24, 1.16, chr(97 + k), transform=ax.transAxes, fontweight='bold', fontsize=8, va='top')
        if k % ncol == 0: ax.set_ylabel(f'Predicted {unid}')
        if k >= len(chaves) - ncol: ax.set_xlabel(f'Observed {unid}')
    for ax in list(axs.flat)[len(chaves):]: ax.axis('off')
    h, l = axs.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=len(l), bbox_to_anchor=(0.5, -0.01), handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.03, 1, 1), w_pad=0.6, h_pad=1.4)
    salvar(fig, pasta, nome or f'Fig_obs_pred_{alvo}'); return fig


def fig_carbono(d, pasta):
    """Boxplot por sistema do teor (COT), do estoque e do COP. Com uma camada só, o eixo x é o sistema."""
    import matplotlib.pyplot as plt, seaborn as sns
    estilo_publicacao()
    vs = [(v, ROTULOS[v]) for v in ['COT', 'COP', 'COM'] if v in d.columns and d[v].notna().any()]
    fig, ax = plt.subplots(1, len(vs), figsize=(183 * MM, 62 * MM))
    ordem = [t for t in ['MN', 'ILP8', 'PM'] if t in set(d.Tratamento)]
    for a, (v, lab) in zip(np.atleast_1d(ax), vs):
        if d.Camada.nunique() == 1:
            ordem = [t for t in ['MN', 'ILP8', 'PM'] if d.loc[d.Tratamento == t, v].notna().any()]
            bp = a.boxplot([d.loc[d.Tratamento == t, v].dropna() for t in ordem], positions=range(len(ordem)),
                           widths=0.55, showfliers=False, patch_artist=True,
                           medianprops=dict(color='black', linewidth=0.8),
                           whiskerprops=dict(linewidth=0.6), capprops=dict(linewidth=0.6))
            for caixa, t in zip(bp['boxes'], ordem):
                caixa.set_facecolor(CORES[t]); caixa.set_alpha(0.35); caixa.set_linewidth(0.6)
            for i, t in enumerate(ordem):
                q = d.loc[d.Tratamento == t, v].dropna()
                a.scatter(i + np.random.default_rng(i).uniform(-0.12, 0.12, len(q)), q, s=7, marker=MARCAS[t],
                          color=CORES[t], edgecolor='white', linewidth=0.3, zorder=3)
            a.set_xticks(range(len(ordem))); a.set_xticklabels([NOMES[t].replace(' ', '\n') for t in ordem])
            a.set_xlabel('')
        else:
            sns.boxplot(data=d, x='Camada', y=v, hue='Tratamento', order=CAMADAS, palette=CORES, ax=a, fliersize=0)
            a.set_xlabel('Layer')
        a.set_ylabel(lab)
    for k, a in enumerate(np.atleast_1d(ax)):
        a.text(-0.28, 1.04, chr(97 + k), transform=a.transAxes, fontweight='bold', fontsize=8, va='top')
    fig.tight_layout(w_pad=1.2); salvar(fig, pasta, 'Fig_carbono_sistemas'); return fig


def fig_pca(res, pasta):
    """Biplot: pontos por sistema com elipse de 95%, vetores das variáveis (carbono em preto, índices em cinza)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    estilo_publicacao()
    sc, ld, v, pm = res['scores'], res['loadings'], res['var'], res['permanova']
    fig, ax = plt.subplots(figsize=(120 * MM, 100 * MM))
    for t in [t for t in ['MN', 'ILP8', 'PM'] if t in set(sc.Tratamento)]:
        q = sc[sc.Tratamento == t]
        ax.scatter(q.PC1, q.PC2, s=12, marker=MARCAS[t], color=CORES[t], edgecolor='white', linewidth=0.4, label=NOMES[t], zorder=3)
        cov = np.cov(q.PC1, q.PC2); val, vec = np.linalg.eigh(cov); ang = np.degrees(np.arctan2(*vec[:, 1][::-1]))
        ax.add_patch(Ellipse((q.PC1.mean(), q.PC2.mean()), *(2 * np.sqrt(5.991 * val[::-1])), angle=ang,
                             fill=False, lw=0.7, color=CORES[t], zorder=2))
    esc = 0.8 * np.abs(sc[['PC1', 'PC2']].values).max() / np.abs(ld.values).max()
    nomes = {'S2s_BSI_seca': 'S2 BSI dry', 'S2s_NDVI_seca': 'S2 NDVI dry', 'S2s_NDVI_chuva': 'S2 NDVI wet',
             'S2s_freq_solo_exposto': 'S2 bare-soil freq.', 'S2s_NDVI_dp': 'S2 NDVI s.d.'}
    rot = {v_: nomes.get(v_) or v_.replace('_0-10', '').replace('COP', 'POC').replace('COM', 'MAOC').replace('EST', 'SOC stock')
           .replace('DR_', 'UAV ').replace('PL_', 'PS ') for v_ in ld.index}
    # posição dos rótulos: além da ponta do vetor; rótulos do mesmo lado afastados na vertical
    lab = pd.DataFrame({'x': ld.PC1 * esc * 1.12, 'y': ld.PC2 * esc * 1.12})
    m = np.abs(np.r_[sc[['PC1', 'PC2']].values.ravel(), lab.values.ravel()]).max()
    passo = 0.075 * m
    for lado in (lab.x >= 0, lab.x < 0):
        idx = lab[lado].sort_values('y').index
        for i in range(1, len(idx)):
            if lab.loc[idx[i], 'y'] - lab.loc[idx[i - 1], 'y'] < passo:
                lab.loc[idx[i], 'y'] = lab.loc[idx[i - 1], 'y'] + passo
    # limites: cabem os pontos, os vetores e o texto dos rótulos (largura aproximada pelo nº de letras)
    larg = {v_: 0.045 * m * len(rot[v_]) for v_ in ld.index}
    xmax = max(m, max(lab.x[v_] + larg[v_] for v_ in ld.index if lab.x[v_] >= 0)) * 1.04
    xmin = min(-m, min(lab.x[v_] - larg[v_] for v_ in ld.index if lab.x[v_] < 0)) * 1.04
    ymax, ymin = max(m, lab.y.max()) * 1.08, min(-m, lab.y.min()) * 1.08
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_aspect('equal')
    for var, (a, b) in ld.iterrows():
        carb = var.startswith(('COP', 'EST', 'COT', 'COM'))
        cor = '0.1' if carb else '0.5'
        ax.annotate('', xy=(a * esc, b * esc), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='-|>', lw=0.8 if carb else 0.5, color=cor, mutation_scale=5, shrinkA=0, shrinkB=0))
        ax.text(lab.x[var], lab.y[var], rot[var], fontsize=5.5, color=cor, fontweight='bold' if carb else 'normal',
                ha='left' if a >= 0 else 'right', va='center', zorder=4,
                bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none', alpha=0.8))
    ax.axhline(0, color='0.8', lw=0.5, zorder=0); ax.axvline(0, color='0.8', lw=0.5, zorder=0)
    ax.set_xlabel(f'PC1 ({100*v[0]:.1f}%)'); ax.set_ylabel(f'PC2 ({100*v[1]:.1f}%)')
    ax.text(0.98, 0.02, f'PERMANOVA: pseudo-F = {pm["F"]:.1f}\n$R^2$ = {pm["R2"]:.2f}, P = {pm["p"]:.4f}', transform=ax.transAxes, fontsize=5.5, va='bottom', ha='right')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, handletextpad=0.3)
    fig.tight_layout(); salvar(fig, pasta, 'Fig_PCA_biplot'); return fig


NOMES_VAR = {'S2s_NDVI_seca': 'NDVI, dry season', 'S2s_NDVI_chuva': 'NDVI, wet season', 'S2s_NDVI_dp': 'NDVI s.d.',
             'S2s_BSI_seca': 'Bare-soil index, dry season', 'S2s_freq_solo_exposto': 'Bare-soil frequency'}


def fig_mapa(g, w, pasta, alvo='COP_0-10'):
    """Dois painéis: COP predito e incerteza, 10 m, com os pontos amostrados (cor = observado),
    extrapolação hachurada, barra de escala e norte."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    estilo_publicacao()
    E0, N0 = g.E.min(), g.N.min()
    fig, axs = plt.subplots(1, 2, figsize=(183 * MM, 92 * MM), sharey=True)
    obs = w.dropna(subset=[alvo])
    vmax = np.nanpercentile(np.r_[g.pred, obs[alvo]], 98)
    for k, (ax, col, cmap, lab) in enumerate(zip(axs, ['pred', 'incerteza'], ['YlGn', 'Greys'],
                                                 [f'Predicted {ROTULOS["COP"]}', 'Uncertainty (s.d. among trees)'])):
        norm = Normalize(0, vmax if col == 'pred' else np.nanpercentile(g[col], 98))
        im = ax.scatter((g.E - E0) / 1000, (g.N - N0) / 1000, c=g[col], cmap=cmap, norm=norm, s=0.6, marker='s', linewidths=0, rasterized=True)
        ex = g[g.extrapola]
        ax.scatter((ex.E - E0) / 1000, (ex.N - N0) / 1000, s=0.6, marker='x', color='#c0392b', linewidths=0.2, rasterized=True)
        for t in ['MN', 'ILP8', 'PM']:
            q = obs[obs.Tratamento == t]
            ax.scatter((q.E - E0) / 1000, (q.N - N0) / 1000, s=10, marker=MARCAS[t], **(dict(c=q[alvo].values, cmap=cmap, norm=norm) if col == 'pred' else dict(facecolor='none')), edgecolor='k', linewidth=0.5, zorder=3, label=NOMES[t])
        # nome de cada área fora dela: MN abaixo, PM acima, ILP8 à direita
        for t, (ox, oy, ha) in {'MN': (0, -0.18, 'center'), 'PM': (0, 0.12, 'center'), 'ILP8': (-0.3, 0, 'right')}.items():
            q = obs[obs.Tratamento == t]
            if len(q) == 0: continue
            xx = (q.E.mean() - E0) / 1000 + ox if t == 'ILP8' else (q.E.mean() - E0) / 1000
            yy = (q.N.mean() - N0) / 1000 if t == 'ILP8' else ((q.N.min() if oy < 0 else q.N.max()) - N0) / 1000 + oy
            ax.text(xx, yy, NOMES[t], ha=ha, va='center', fontsize=6, fontweight='bold',
                    bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none', alpha=0.85))
        cb = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.045, pad=0.12, aspect=30); cb.set_label(lab); cb.outline.set_linewidth(0.4)
        ax.set_aspect('equal'); ax.set_xlabel('Easting (km)')
        ax.text(-0.08, 1.04, chr(97 + k), transform=ax.transAxes, fontweight='bold', fontsize=8)
        x0, y0 = ax.get_xlim()[1] - 0.65, ax.get_ylim()[1] - 0.75          # escala de 500 m e norte, canto superior direito (vazio)
        ax.plot([x0, x0 + 0.5], [y0, y0], color='k', lw=1.5); ax.text(x0 + 0.25, y0 + 0.06, '500 m', ha='center', fontsize=5.5)
        ax.annotate('N', xy=(x0 + 0.45, y0 + 0.55), xytext=(x0 + 0.45, y0 + 0.3), ha='center', fontsize=6,
                    arrowprops=dict(arrowstyle='-|>', lw=0.6, color='k', mutation_scale=6))
    axs[0].set_ylabel('Northing (km)')
    fig.text(0.5, 0.005, 'UTM zone 23S (SIRGAS 2000), origin at the lower-left pixel; × = covariates outside the calibration range',
             ha='center', fontsize=5.5, color='0.35')
    fig.tight_layout(rect=(0, 0.02, 1, 1)); salvar(fig, pasta, f'Fig_mapa_{alvo}'); return fig


def fig_shap(sh, pasta):
    """a: beeswarm SHAP (cor = valor da variável); b: dependência da frequência de solo exposto, por sistema."""
    import matplotlib.pyplot as plt
    estilo_publicacao()
    S, X = sh['shap'], sh['X']
    ordem = S.abs().mean().sort_values().index
    fig, axs = plt.subplots(1, 2, figsize=(183 * MM, 70 * MM), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axs[0]; rng = np.random.default_rng(1)
    for i, v in enumerate(ordem):
        z = (X[v] - X[v].min()) / (X[v].max() - X[v].min() + 1e-12)
        y = i + rng.uniform(-0.28, 0.28, len(S)) * 0.8
        sc = ax.scatter(S[v], y, c=z, cmap='coolwarm', s=7, linewidths=0, vmin=0, vmax=1)
    ax.set_yticks(range(len(ordem))); ax.set_yticklabels([NOMES_VAR.get(v, v) for v in ordem]); ax.axvline(0, color='0.6', lw=0.5)
    ax.set_xlabel(f'SHAP value (impact on predicted POC, g kg$^{{-1}}$)')
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, ticks=[0, 1]); cb.ax.set_yticklabels(['Low', 'High']); cb.set_label('Feature value'); cb.outline.set_linewidth(0.4)
    ax = axs[1]; v = ordem[-1]   # variável mais importante
    for t in ['MN', 'ILP8', 'PM']:
        m = sh['sis'] == t
        ax.scatter(X[v][m] * (100 if 'freq' in v else 1), S[v][m], s=12, marker=MARCAS[t], color=CORES[t], edgecolor='white', linewidth=0.4, label=NOMES[t])
    ax.axhline(0, color='0.6', lw=0.5); ax.set_xlabel({'S2s_freq_solo_exposto': 'Bare-soil frequency, 2019–2024 (% of clear images)'}.get(v, NOMES_VAR.get(v, v) + ', 2019–2024'))
    ax.set_ylabel('SHAP value (g kg$^{-1}$)'); ax.legend(loc='best')
    for k, a in enumerate(axs):
        a.text(-0.12 if k else -0.42, 1.03, chr(97 + k), transform=a.transAxes, fontweight='bold', fontsize=8)
    fig.tight_layout(w_pad=2); salvar(fig, pasta, f'Fig_SHAP_{sh["alvo"]}'); return fig


def fig_halteres(tab, pasta, alvos=('COT_0-10', 'COP_0-10', 'COM_0-10')):
    """Para cada conjunto (melhor modelo), R² com todos os pontos ligado ao R² dentro dos sistemas.
    A linha mostra quanto do acerto vem só de reconhecer o sistema."""
    import matplotlib.pyplot as plt
    estilo_publicacao()
    alvos = [a for a in alvos if a in set(tab.alvo)]
    ordem = [c for c in CONJ_EN if c in set(tab.conjunto)]
    fig, axs = plt.subplots(1, len(alvos), figsize=(183 * MM, (5 * len(ordem) + 20) * MM), sharey=True)
    tit = {'COP_0-10': 'POC', 'COT_0-10': 'SOC', 'COM_0-10': 'MAOC', 'EST_0-10': 'SOC stock'}
    for k, (ax, a) in enumerate(zip(np.atleast_1d(axs), alvos)):
        x = tab[tab.alvo == a].reset_index(drop=True)
        x = x.loc[x.groupby('conjunto').R2_cv.idxmax()].set_index('conjunto').reindex(ordem)
        y = np.arange(len(ordem))[::-1]
        for yi, (r1, r2) in zip(y, x[['R2_cv', 'R2_dentro']].values):
            ax.plot([r1, r2], [yi, yi], color='0.75', lw=1.2, zorder=1)
        ax.scatter(x.R2_cv, y, s=16, color='0.35', zorder=2, label='All points')
        ax.scatter(x.R2_dentro, y, s=16, color='#1B7837', zorder=3, label='Within systems')
        ax.axvline(0, color='k', lw=0.5); ax.set_title(tit.get(a, a), loc='left', fontweight='bold')
        ax.set_xlabel('LOOCV $R^2$')
        lim0 = max(-1, np.nanmin(tab[tab.alvo.isin(alvos)][['R2_cv', 'R2_dentro']].values) - 0.1); ax.set_xlim(lim0, 1)
        ax.text(-0.06 if k else -0.75, 1.06, chr(97 + k), transform=ax.transAxes, fontweight='bold', fontsize=8)
    axs = np.atleast_1d(axs); axs[0].set_yticks(np.arange(len(ordem))[::-1]); axs[0].set_yticklabels([rotulo_conj(c) for c in ordem])
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.6, 1.0), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1); salvar(fig, pasta, 'Fig_halteres_R2'); return fig


def fig_entre_dentro(t, pasta, alvo):
    import matplotlib.pyplot as plt
    estilo_publicacao()
    x = t[t.alvo == alvo].sort_values('r_total')
    fig, ax = plt.subplots(figsize=(89 * MM, (3.2 * len(x) + 14) * MM))
    y = np.arange(len(x))
    ax.barh(y - .2, x.r_total, .38, label='All points (between + within systems)', color='0.65')
    ax.barh(y + .2, x.r_dentro, .38, label='Within systems', color='#1B7837')
    sig = x.p_dentro_FDR < 0.05
    for yi, r in zip(y[sig.values], x.r_dentro[sig.values]):
        ax.text(r + np.sign(r) * 0.02, yi + .2, '*', va='center', ha='left' if r > 0 else 'right')
    ax.set_yticks(y); ax.set_yticklabels(x['cov']); ax.axvline(0, color='k', lw=.8)
    ax.set_xlabel(f'Pearson r with {alvo} (* within-system FDR < 0.05)'); ax.legend(loc='lower right')
    fig.tight_layout(); salvar(fig, pasta, f'Fig_r_entre_dentro_{alvo}'); return fig


def fig_r2(res, pasta, alvo):
    """R² de validação cruzada (com IC 95%) e R² além da média do sistema, melhor modelo de cada conjunto."""
    import matplotlib.pyplot as plt
    estilo_publicacao()
    x = res[(res.alvo == alvo)].reset_index(drop=True)
    x = x.loc[x.groupby('conjunto').R2_cv.idxmax()].sort_values('R2_cv')
    fig, ax = plt.subplots(figsize=(89 * MM, (6 * len(x) + 18) * MM))
    y = np.arange(len(x))
    ax.barh(y - .2, x.R2_cv, .38, label='$R^2_{cv}$ (all points)', color='0.65')
    if 'R2_IC95_inf' in x:
        ax.errorbar(x.R2_cv, y - .2, xerr=[x.R2_cv - x.R2_IC95_inf, x.R2_IC95_sup - x.R2_cv], fmt='none',
                    ecolor='0.2', elinewidth=0.6, capsize=1.5)
    ax.barh(y + .2, x.R2_dentro, .38, label='$R^2$ beyond system mean', color='#1B7837')
    ax.set_yticks(y); ax.set_yticklabels([rotulo_conj(c, m, 30) for c, m in zip(x.conjunto, x.modelo)]); ax.axvline(0, color='k', lw=.6)
    ax.set_xlabel('LOOCV $R^2$ (error bars: 95% bootstrap CI)')
    ax.legend(loc='lower center', bbox_to_anchor=(0.4, 1.0), ncol=2, handlelength=1.2)
    fig.tight_layout(); salvar(fig, pasta, f'Fig_R2_{alvo}'); return fig


def fig_repetibilidade(w, pasta, vs=('COT', 'COP', 'COM'), camada='0-10'):
    """Seco (x) × chuvoso (y) nos mesmos pontos, um painel por variável: cor e forma por sistema, linha 1:1
    tracejada e, no canto, o r de Pearson de cada sistema e o r dentro dos sistemas (tabela de repetibilidade)."""
    import matplotlib.pyplot as plt
    estilo_publicacao()
    tab = repetibilidade(w, vs, camada)
    vs = [v for v in vs if v in set(tab.variavel)]
    fig, axs = plt.subplots(1, len(vs), figsize=(60 * len(vs) * MM, 76 * MM), squeeze=False)
    for k, (ax, v) in enumerate(zip(axs.flat, vs)):
        a, b = f'{v}_{camada}', f'{v}_{camada}_ch'
        x = w[['Tratamento', a, b]].dropna()
        lo, hi = np.nanmin(x[[a, b]].values), np.nanmax(x[[a, b]].values); pad = 0.06 * (hi - lo); lim = (lo - pad, hi + pad)
        ax.plot(lim, lim, ls=(0, (3, 2)), lw=0.6, color='0.55', zorder=1)
        txt = []
        for t in [t for t in ['MN', 'ILP8', 'PM'] if t in set(x.Tratamento)]:
            q = x[x.Tratamento == t]
            ax.scatter(q[a], q[b], s=12, marker=MARCAS[t], color=CORES[t], edgecolor='white', linewidth=0.4,
                       label=NOMES[t], zorder=3)
            txt.append(f'{NOMES[t]}: r = {tab[(tab.variavel == v) & (tab.sistema == t)].r.iloc[0]:.2f}')
        txt.append(f'Within systems: r = {tab[(tab.variavel == v) & (tab.sistema == "Dentro dos sistemas")].r.iloc[0]:.2f}')
        ax.text(0.0, 1.03, '\n'.join(txt), transform=ax.transAxes, va='bottom', ha='left', fontsize=5.5, linespacing=1.3)   # acima do painel, sem cobrir pontos
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
        unid = ROTULOS.get(v, v)
        ax.set_xlabel(f'{unid}, dry season'); ax.set_ylabel(f'{unid}, wet season')
        ax.text(-0.3, 1.3, chr(97 + k), transform=ax.transAxes, fontweight='bold', fontsize=8, va='top')
    h, l = axs.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=len(l), bbox_to_anchor=(0.5, -0.01), handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=1.5)
    salvar(fig, pasta, f'Fig_seco_chuvoso_{camada}'); return fig


# ----------------------------------------------------------------------------- 22/10/2026: todas as camadas e partição da variância
def letras(medias, pares, alfa=0.05):
    """Letras de significância (compact letter display) para poucos grupos.
    medias: Series sistema -> média; pares: DataFrame com 'par' ('A x B') e 'p_Holm'.
    Grupos que não diferem compartilham letra; a letra 'a' vai para a maior média."""
    ordem = list(medias.sort_values(ascending=False).index)
    dif = {frozenset(p.split(' x ')): q < alfa for p, q in zip(pares.par, pares.p_Holm)}
    nd = lambda a, b: not dif.get(frozenset((a, b)), False)
    grupos = []
    for i, t in enumerate(ordem):            # cada grupo máximo de sistemas que não diferem entre si
        g = [t] + [u for u in ordem[i + 1:] if nd(t, u)]
        g = [u for u in g if all(nd(u, v) for v in g if v != u)]
        if not any(set(g) <= set(h) for h in grupos):
            grupos.append(g)
    out = {t: '' for t in ordem}
    for k, g in enumerate(grupos):
        for t in g:
            out[t] += 'abcdefgh'[k]
    return out


def fig_carbono_camadas(d, comp, pasta, nome='Fig_carbono_sistemas_camadas'):
    """Teor de COT, estoque e COP por camada (eixo x) e sistema (cores), com letras de Holm dentro de cada camada.
    Letras iguais: sistemas que não diferem (Mann-Whitney com correção de Holm, p ≥ 0,05)."""
    import matplotlib.pyplot as plt
    estilo_publicacao()
    vs = [(v, ROTULOS[v]) for v in ['COT', 'COP', 'COM'] if v in d.columns and d[v].notna().any()]
    cams = [c for c in CAMADAS if c in set(d.Camada)]
    sis = [t for t in ['MN', 'ILP8', 'PM'] if t in set(d.Tratamento)]
    fig, axs = plt.subplots(1, len(vs), figsize=(183 * MM, 66 * MM))
    larg = 0.24
    for k, (ax, (v, lab)) in enumerate(zip(np.atleast_1d(axs), vs)):
        topo = np.nanmax(d[v]) * 1.02
        for j, c in enumerate(cams):
            q = d[d.Camada == c]
            pares = comp[v]['pares']; pares = pares[pares.Camada == c]
            med = q.groupby('Tratamento')[v].mean()
            le = letras(med, pares) if len(pares) else {t: '' for t in sis}
            todas_iguais = len(set(le.values())) == 1
            for i, t in enumerate(sis):
                y = q.loc[q.Tratamento == t, v].dropna(); x0 = j + (i - 1) * larg
                bp = ax.boxplot([y], positions=[x0], widths=larg * 0.8, showfliers=False, patch_artist=True,
                                medianprops=dict(color='black', linewidth=0.7), whiskerprops=dict(linewidth=0.5),
                                capprops=dict(linewidth=0.5))
                bp['boxes'][0].set_facecolor(CORES[t]); bp['boxes'][0].set_alpha(0.35); bp['boxes'][0].set_linewidth(0.5)
                ax.scatter(x0 + np.random.default_rng(10 * j + i).uniform(-0.05, 0.05, len(y)), y, s=4, marker=MARCAS[t],
                           color=CORES[t], edgecolor='white', linewidth=0.2, zorder=3, label=NOMES[t] if j == 0 else None)
                if not todas_iguais:
                    ax.text(x0, y.max() + 0.02 * topo, le[t], ha='center', va='bottom', fontsize=6)
            if todas_iguais:
                ax.text(j, q[v].max() + 0.04 * topo, 'ns', ha='center', va='bottom', fontsize=6, color='0.35')
        ax.set_xticks(range(len(cams))); ax.set_xticklabels([c.replace('cm', ' cm') for c in cams])
        ax.set_xlabel('Soil layer'); ax.set_ylabel(lab); ax.set_ylim(top=topo * 1.12)
        ax.text(-0.25, 1.04, chr(97 + k), transform=ax.transAxes, fontweight='bold', fontsize=8, va='top')
    h, l = np.atleast_1d(axs)[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, markerscale=2)
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=1.2); salvar(fig, pasta, nome); return fig


def particao_variancia(w, alvos, n_boot=2000, seed=1):
    """Fração da variância de cada variável que está ENTRE os sistemas (eta² = SQ entre / SQ total),
    com IC 95% por bootstrap estratificado (reamostra pontos dentro de cada sistema).
    É o R² máximo que um modelo que só conhece o sistema consegue; o restante (1 - eta²) é a variação
    dentro dos sistemas, a que interessa para mapear o carbono no talhão."""
    rng = np.random.default_rng(seed)
    lin = []
    for a in alvos:
        if a not in w.columns:
            continue
        x = w[['Tratamento', a]].dropna()
        gs = [g[a].values for _, g in x.groupby('Tratamento')]
        def eta2(gs):
            tudo = np.concatenate(gs); m = tudo.mean()
            sst = ((tudo - m) ** 2).sum(); ssb = sum(len(g) * (g.mean() - m) ** 2 for g in gs)
            return ssb / sst
        e = eta2(gs)
        bs = [eta2([g[rng.integers(0, len(g), len(g))] for g in gs]) for _ in range(n_boot)]
        cv_dentro = np.mean([g.std(ddof=1) / g.mean() * 100 for g in gs])
        lin.append(dict(alvo=a, n=len(x), eta2=e, eta2_IC95_inf=np.percentile(bs, 2.5), eta2_IC95_sup=np.percentile(bs, 97.5),
                        CV_dentro_medio_pct=cv_dentro))
    return pd.DataFrame(lin)


def tabela_inflacao(tab, part, conj_rs=('Drone', 'Planet', 'Sentinel-2 (data)', 'Sentinel-2 (série)', 'Sentinel-2 + relevo + textura')):
    """Para cada variável: eta² (variância entre sistemas), R² do modelo que só conhece o sistema,
    melhor R²cv dos conjuntos de sensoriamento (sem o sistema como covariável) e o R² dentro dos sistemas desse modelo."""
    lin = []
    for a, q in tab.groupby('alvo'):
        p = part[part.alvo == a]
        if p.empty:
            continue
        so = q[q.conjunto == 'Só o sistema'].R2_cv.max()
        r = q[q.conjunto.isin(conj_rs)]
        if r.empty:
            continue
        b = r.loc[r.R2_cv.idxmax()]
        lin.append(dict(alvo=a, eta2=p.eta2.iloc[0], eta2_IC95_inf=p.eta2_IC95_inf.iloc[0], eta2_IC95_sup=p.eta2_IC95_sup.iloc[0],
                        R2_so_sistema=so, melhor_conjunto=b.conjunto, modelo=b.modelo, R2_cv=b.R2_cv, R2_dentro=b.R2_dentro))
    return pd.DataFrame(lin)


def fig_inflacao(ti, pasta, nome='Fig_inflacao_R2'):
    """a: R²cv do melhor modelo de sensoriamento contra a fração da variância entre sistemas (eta²), uma variável
    por ponto (teor, estoque e COP nas camadas). Pontos sobre a linha 1:1 = o acerto vem do contraste entre sistemas.
    b: R² dentro dos sistemas das mesmas variáveis (o que sobra para o mapeamento no talhão)."""
    import matplotlib.pyplot as plt
    estilo_publicacao()
    cor = {'COT': '#5E3C99', 'COP': '#1B7837', 'COM': '#E69F00'}
    mar = {'0-10': 'o', '10-20': 's', '20-30': '^', '0-30': 'D'}
    nome_v = {'COT': 'SOC', 'COP': 'POC', 'COM': 'MAOC'}
    fig, axs = plt.subplots(1, 2, figsize=(183 * MM, 80 * MM), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axs[0]
    lo = min(-0.3, np.nanmin(ti[['R2_cv', 'R2_dentro']].values) - 0.05)
    ax.plot([0, 1], [0, 1], ls=(0, (3, 2)), color='0.5', lw=0.7, zorder=1)
    ax.axhline(0, color='0.8', lw=0.5, zorder=0)
    for _, r in ti.iterrows():
        v, c = r.alvo.split('_', 1)
        ax.errorbar(r.eta2, r.R2_cv, xerr=[[r.eta2 - r.eta2_IC95_inf], [r.eta2_IC95_sup - r.eta2]], fmt='none',
                    ecolor=cor[v], elinewidth=0.5, alpha=0.6, zorder=2)
        ax.scatter(r.eta2, r.R2_cv, s=22, marker=mar.get(c, 'o'), color=cor[v], edgecolor='white', linewidth=0.4, zorder=3)
    ax.set_xlim(-0.02, 1); ax.set_ylim(lo, 1)
    ax.set_xlabel('Variance between systems ($\\eta^2$)'); ax.set_ylabel('Best remote sensing model, LOOCV $R^2$')
    ax.text(0.97, 0.9, '1:1', transform=ax.transAxes, ha='right', fontsize=6, color='0.4')
    ax = axs[1]
    ti2 = ti.sort_values('eta2')
    y = np.arange(len(ti2))
    for yi, (_, r) in zip(y, ti2.iterrows()):
        v, c = r.alvo.split('_', 1)
        ax.plot([r.R2_dentro, r.R2_cv], [yi, yi], color='0.75', lw=1.0, zorder=1)
        ax.scatter(r.R2_cv, yi, s=16, color='0.35', marker=mar.get(c, 'o'), zorder=2)
        ax.scatter(r.R2_dentro, yi, s=16, color=cor[v], marker=mar.get(c, 'o'), zorder=3)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_yticks(y); ax.set_yticklabels([f"{nome_v[a.split('_')[0]]}, {a.split('_')[1]} cm" for a in ti2.alvo])
    ax.set_xlabel('LOOCV $R^2$: all points (grey) and within systems (colour)'); ax.set_xlim(lo, 1)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls='', marker='o', color=cor[k], label=nome_v[k]) for k in cor] + \
        [Line2D([], [], ls='', marker=mar[k], color='0.4', label=f'{k} cm') for k in mar if any(ti.alvo.str.endswith(k))]
    fig.legend(handles=h, loc='upper center', ncol=len(h), bbox_to_anchor=(0.5, 1.0), frameon=False, handletextpad=0.2, columnspacing=1)
    for k, a in enumerate(axs):
        a.text(-0.15 if k == 0 else -0.45, 1.03, chr(97 + k), transform=a.transAxes, fontweight='bold', fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.5); salvar(fig, pasta, nome); return fig

