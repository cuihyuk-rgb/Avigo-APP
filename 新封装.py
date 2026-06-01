import os, json, pickle, math
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor
import shap
ASSETS_DIR = 'assets'
AIRPORT_DB = 'world-airports.csv'
st.set_page_config(page_title='AviGo 包机价格预测', page_icon='✈️', layout='centered')
# =========================== 特征名汉化映射 ===========================
FEATURE_NAME_ZH = {
    'seller_company': '销售公司',
    'category': '机型分类',
    'type': '飞机型号',
    'aircraft_base': '飞机基地',
    'itinerary_from': '出发机场',
    'itinerary_to': '目的机场',
    'safety': '安全认证',
    'route': '航线',
    'dist_num': '飞行距离',
    'leg_min': '飞行时间',
    'pos_min': '调机时间',
    'seat': '座位数',
    'age': '飞机年龄',
    'pos_ratio': '调机/飞行比',
    'total_time': '总时长',
    'r_avg': '航线历史均价',
    'r_cnt': '航线热度',
    'b_avg': '基地历史均价',
    'b_cnt': '基地热度',
    'type_avg_pos': '本机型平均调机时间',
    'type_avg_price': '本机型平均价格',
    'amen_wifi': 'WiFi',
    'amen_phone': '卫星电话',
    'amen_toilet': '独立洗手间',
    'amen_pets': '允许宠物',
    'amen_smoking': '允许吸烟',
    'amen_tv': '电视',
    'amen_attendant': '空乘',
}
# =========================== 缓存加载 ===========================
@st.cache_resource
def load_assets():
    models = {}
    for name in ['P05', 'P50', 'P95']:
        m = CatBoostRegressor()
        m.load_model(os.path.join(ASSETS_DIR, f'model_v2_{name}.cbm'))
        models[name] = m
    with open(os.path.join(ASSETS_DIR, 'calibration.json')) as f:
        cal = json.load(f)
    with open(os.path.join(ASSETS_DIR, 'feature_config.json')) as f:
        feat_cfg = json.load(f)
    with open(os.path.join(ASSETS_DIR, 'statistics.pkl'), 'rb') as f:
        stats = pickle.load(f)
    with open(os.path.join(ASSETS_DIR, 'airport_coords.json')) as f:
        coords = {k: tuple(v) for k, v in json.load(f).items()}
    airports_db = pd.read_csv(AIRPORT_DB, low_memory=False)
    airports_db = airports_db[airports_db['ident'].str.len() == 4]
    airports_db = airports_db[airports_db['ident'].isin(coords.keys())]
    airports_db = airports_db[['ident', 'name', 'municipality', 'iso_country']].dropna(subset=['ident'])
    airports_db['display'] = (
        airports_db['ident'] + ' - ' +
        airports_db['name'].fillna('') + ', ' +
        airports_db['municipality'].fillna('')
    )
    airports_db = airports_db.sort_values('ident')
    type_info_path = os.path.join(ASSETS_DIR, 'type_info.json')
    if os.path.exists(type_info_path):
        with open(type_info_path) as f:
            type_to_category = json.load(f)
    else:
        type_to_category = {t: 'Unknown' for t in stats['type_price'].keys()}
    category_types = {}
    for t, cat in type_to_category.items():
        category_types.setdefault(cat, []).append(t)
    for cat in category_types:
        category_types[cat] = sorted(category_types[cat])
    categories = sorted(category_types.keys())
    explainer = shap.TreeExplainer(models['P50'])
    return (models, cal, feat_cfg, stats, coords,
            airports_db, categories, category_types, type_to_category, explainer)
(models, cal, feat_cfg, stats, coords,
 airports_db, categories, category_types, type_to_category, explainer) = load_assets()
# =========================== 工具函数 ===========================
def haversine(lat1, lon1, lat2, lon2):
    r = 3440.065
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return r * 2 * math.asin(math.sqrt(a))

def search_airport(keyword, top_n=30):
    if not keyword:
        return airports_db.head(top_n)
    k = keyword.lower()
    mask = (airports_db['ident'].str.lower().str.contains(k) |
            airports_db['name'].str.lower().str.contains(k, na=False) |
            airports_db['municipality'].str.lower().str.contains(k, na=False))
    return airports_db[mask].head(top_n)

def build_features(from_icao, to_icao, category, a_type, seller, seat_val, age_val, amen_inputs):
    lat1, lon1 = coords[from_icao]
    lat2, lon2 = coords[to_icao]
    dist_nm = haversine(lat1, lon1, lat2, lon2)
    flt_min = dist_nm / 450 * 60
    route = f"{from_icao}→{to_icao}"
    r_avg = stats['route_avg'].get(route, stats['gbl_price'])
    r_cnt = stats['route_cnt'].get(route, 0)
    t_pos = stats['type_pos'].get(a_type, stats['gbl_pos'])
    t_prc = stats['type_price'].get(a_type, stats['gbl_price'])

    inp = {
        'category': category,
        'type': a_type,
        'itinerary_from': from_icao,
        'itinerary_to': to_icao,
        'seller_company': seller,
        'safety': 'Unknown',
        'route': route,
        'dist_num': dist_nm,
        'leg_min': flt_min,
        'seat': float(seat_val) if seat_val else stats['gbl_seat'],
        'age': float(age_val) if age_val else stats['gbl_age'],
        'r_avg': r_avg,
        'r_cnt': r_cnt,
        'type_avg_pos': t_pos,
        'type_avg_price': t_prc,
    }
    for a in feat_cfg['amenities']:
        inp[f'amen_{a}'] = int(amen_inputs.get(a, False))

    df_in = pd.DataFrame([inp])[feat_cfg['features']]
    return df_in, dist_nm, flt_min, route, inp

def predict_price(df_in):
    p05 = float(models['P05'].predict(df_in)[0])
    p50 = float(models['P50'].predict(df_in)[0])
    p95 = float(models['P95'].predict(df_in)[0])
    lower = max(550, p05 + cal['p05_offset'], p50 * 0.25)
    upper = min(1_000_000, p95 - cal['p95_offset'], p50 * 4.0)
    return round(lower, -2), round(p50, -2), round(upper, -2)

def get_shap_contributions(df_in):
    shap_values = explainer.shap_values(df_in)
    base_value = explainer.expected_value
    contributions = []
    for i, feat in enumerate(df_in.columns):
        val = df_in[feat].iloc[0]
        contrib_dollar = int(round(shap_values[0][i]))
        zh_name = FEATURE_NAME_ZH.get(feat, feat)
        if isinstance(val, float):
            val_display = f"{val:,.2f}"
        else:
            val_display = str(val)
        contributions.append({
            '特征名': zh_name,
            '特征原值': val_display,
            '贡献金额': contrib_dollar,
        })
    contributions.sort(key=lambda x: abs(x['贡献金额']), reverse=True)
    return base_value, contributions

def filter_categories_by_distance(dist_nm):
    if dist_nm <= 1500:
        return ['Light Jet', 'Midsize Jet']
    elif dist_nm <= 3000:
        return ['Light Jet', 'Midsize Jet', 'Super Midsize Jet']
    else:
        return ['Super Midsize Jet', 'Heavy Jet', 'Ultra Long Range', 'VIP Airliner']
# =========================== 机场选择组件 ===========================
def airport_selector(label, key_prefix):
    icao_key = f'{key_prefix}_icao'
    display_key = f'{key_prefix}_display'
    if icao_key not in st.session_state:
        st.session_state[icao_key] = None
    if display_key not in st.session_state:
        st.session_state[display_key] = None

    if st.session_state[icao_key]:
        st.success(f'✅ {label}: {st.session_state[display_key]}')
        if st.button(f'🔄 更换{label}', key=f'change_{key_prefix}'):
            st.session_state[icao_key] = None
            st.session_state[display_key] = None
            st.rerun()
        return st.session_state[icao_key]
    else:
        kw = st.text_input(f'搜索{label} (城市/机场/ICAO)', key=f'{key_prefix}_kw', placeholder='如 New York, KJFK')
        results = search_airport(kw)
        if not results.empty:
            labels = results['display'].tolist()
            idents = results['ident'].tolist()
            pick = st.selectbox(f'选择{label}', labels, key=f'{key_prefix}_pick')
            if st.button(f'✅ 确认{label}', key=f'confirm_{key_prefix}', use_container_width=True):
                idx = labels.index(pick)
                st.session_state[icao_key] = idents[idx]
                st.session_state[display_key] = pick
                st.rerun()
        return None
# =========================== 主界面 ===========================
st.title('✈️ AviGo 包机价格预测')
st.caption('输入行程与机型，获取智能价格参考与价格构成分析')
mode = st.sidebar.radio('选择功能模式', ['单次精准报价', '多方案比较'])
# =========================== 模式 1 ===========================
if mode == '单次精准报价':
    st.subheader('🛫🛬 出发 & 目的机场')
    col_left, col_right = st.columns(2)
    with col_left:
        from_icao = airport_selector('出发机场', 'from')
    with col_right:
        to_icao = airport_selector('目的机场', 'to')
    st.divider()
    with st.form('predict_form'):
        st.subheader('✈️ 飞机选择')
        col_cat, col_type = st.columns(2)
        with col_cat:
            selected_category = st.selectbox('机型分类', categories, index=None, placeholder='选择类别')
        with col_type:
            if selected_category and selected_category in category_types:
                selected_type = st.selectbox('飞机型号', category_types[selected_category], placeholder='选择具体型号')
            else:
                selected_type = None
                st.caption('请先选择类别')
        st.markdown('**附加信息（选填）**')
        col1, col2, col3 = st.columns(3)
        with col1:
            seller = st.text_input('销售公司', placeholder='如 NetJets')
        with col2:
            seat_val = st.number_input('座位数', min_value=1, max_value=100, value=None, placeholder='自动')
        with col3:
            age_val = st.number_input('机龄(年)', min_value=0.0, max_value=60.0, value=None, step=0.5, placeholder='自动')
        st.markdown('**机上设施（可选）**')
        amen_cols = st.columns(7)
        amen_inputs = {}
        amen_labels = {
            'wifi': '📶 WiFi', 'phone': '📞 卫星电话', 'toilet': '🚻 洗手间',
            'pets': '🐾 宠物', 'smoking': '🚬 吸烟', 'tv': '📺 电视',
            'attendant': '👩‍✈️ 空乘'
        }
        for i, (key, label) in enumerate(amen_labels.items()):
            with amen_cols[i]:
                amen_inputs[key] = st.checkbox(label)
        submitted = st.form_submit_button('💰 获取价格预估', type='primary', use_container_width=True)
    if submitted:
        errors = []
        if not from_icao: errors.append('请选择出发机场')
        if not to_icao: errors.append('请选择目的机场')
        if not selected_category or not selected_type:
            errors.append('请选择机型分类和具体型号')
        if errors:
            for e in errors:
                st.error(f'⚠️ {e}')
        else:
            with st.spinner('正在计算价格...'):
                df_in, dist_nm, flt_min, route, raw_input = build_features(
                    from_icao, to_icao, selected_category, selected_type,
                    seller.strip() or 'Unknown', seat_val, age_val, amen_inputs
                )
                lower, median, upper = predict_price(df_in)
                base_value, contributions = get_shap_contributions(df_in)
            st.divider()
            st.subheader('📊 预估结果')
            c1, c2, c3 = st.columns(3)
            c1.metric('📉 价格下限', f'${lower:,.0f}')
            with c2:
                st.metric('💵 参考价格', f'${median:,.0f}')
                with st.popover('📋 查看价格详情', use_container_width=True):
                    st.markdown('### 🔍 价格影响因素明细')
                    st.caption(f'市场基准价（所有特征取均值时）约为 **${base_value:,.0f}**，以下因素对最终价格产生调整：')
                    for item in contributions[:10]:
                        feat = item['特征名']
                        val = item['特征原值']
                        contrib = item['贡献金额']
                        if contrib >= 0:
                            color = '#28a745'
                            sign = '+'
                        else:
                            color = '#dc3545'
                            sign = '-'
                        st.markdown(
                            f"<span style='color:{color};font-weight:bold;'>"
                            f"{sign}${abs(contrib):,}</span> "
                            f"{feat} "
                            f"<small style='color:gray;'>(当前: {val})</small>",
                            unsafe_allow_html=True
                        )
                    st.caption('* 以上分析基于模型 P50，绿色表示推高价格，红色表示拉低价格。')

            c3.metric('📈 价格上限', f'${upper:,.0f}')

            st.caption(
                f'🛫 航线: {route} | '
                f'📏 距离: {dist_nm:,.0f} 海里 | '
                f'⏱️ 飞行时间: ~{flt_min:,.0f} 分钟'
            )
# =========================== 模式 2 ===========================
else:
    st.subheader('🛫🛬 行程信息')
    col_left, col_right = st.columns(2)
    with col_left:
        from_icao = airport_selector('出发机场', 'from')
    with col_right:
        to_icao = airport_selector('目的机场', 'to')

    travel_date = st.date_input('出行日期（可选，暂不影响预测）', value=None)

    if st.button('🔍 获取所有可行方案', type='primary', use_container_width=True):
        errors = []
        if not from_icao: errors.append('请选择出发机场')
        if not to_icao: errors.append('请选择目的机场')
        if errors:
            for e in errors:
                st.error(f'⚠️ {e}')
        else:
            with st.spinner('正在分析航线并生成方案...'):
                dist_nm = haversine(*coords[from_icao], *coords[to_icao])
                suitable_cats = filter_categories_by_distance(dist_nm)
                candidate_types = []
                for cat in suitable_cats:
                    if cat in category_types:
                        for t in category_types[cat]:
                            candidate_types.append((cat, t))

                if not candidate_types:
                    st.error('没有适合该航程的机型。')
                else:
                    results = []
                    for cat, t in candidate_types:
                        df_in, _, flt_min, route, _ = build_features(
                            from_icao, to_icao, cat, t, 'Unknown',
                            None, None, {}
                        )
                        lower, median, upper = predict_price(df_in)
                        results.append({
                            '飞机分类': cat,
                            '飞机型号': t,
                            '参考价格 (P50)': f'${median:,.0f}',
                            '价格下限 (P05)': f'${lower:,.0f}',
                            '价格上限 (P95)': f'${upper:,.0f}',
                            '估计座位': int(stats.get('gbl_seat', 0)),
                            '飞行时间 (分钟)': f'{flt_min:.0f}',
                        })

                    df_res = pd.DataFrame(results)
                    st.success(f'找到 {len(df_res)} 个可行方案')
                    st.dataframe(df_res, hide_index=True, use_container_width=True)
                    st.caption('* 价格基于历史模型预估，实际报价可能因具体设施、销售公司等因素变化。')

st.divider()
st.caption('*价格基于历史市场数据预测，实际报价以运营商确认为准 · Powered by AviGo*')