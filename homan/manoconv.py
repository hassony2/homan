#!/usr/bin/env pytho="extra_data/mano/MANO_LEFT.pkl"n
# -*- coding: utf-8 -*-
# pylint: disable=redefined-outer-name,import-error

import pickle

import numpy as np


def left_from_right(right_path="extra_data/mano/MANO_RIGHT.pkl",
                    left_path="extra_data/mano/MANO_LEFT.pkl",
                    invert_path="local_data/manoinvert.pkl"):
    """
    Create MANO_LEFT.pkl from MANO_RIGHT.pkl file
    """
    with open(right_path, "rb") as p_f:
        m_r = pickle.load(p_f, encoding="latin1")
    m_l = {}
    hc_r = m_r["hands_components"]
    hc_l = hc_r.copy()
    hc_l[:, 1::3] = -hc_l[:, 1::3]
    hc_l[:, 2::3] = -hc_l[:, 2::3]
    m_l["hands_components"] = hc_l
    m_l["f"] = m_r["f"][:, ::-1].copy()
    m_l["kintree_table"] = m_r["kintree_table"]
    m_l["J_regressor"] = m_r["J_regressor"]
    m_l["bs_style"] = m_r["bs_style"]
    m_l["bs_type"] = m_r["bs_type"]
    m_l["hands_coeffs"] = m_r["hands_coeffs"]
    m_l["weights"] = m_r["weights"]
    m_l["shapedirs"] = m_r["shapedirs"]
    j_l = m_r["J"]
    j_l[:, 0] = -j_l[:, 0]
    m_l["J"] = j_l

    hm_r = m_r["hands_mean"]
    hm_l = hm_r.copy()
    hm_l[1::3] = -hm_l[1::3]
    hm_l[2::3] = -hm_l[2::3]
    m_l["hands_mean"] = hm_l

    v_l = m_r["v_template"]
    v_l[:, 0] = -v_l[:, 0]
    m_l["v_template"] = v_l

    with open(invert_path, "rb") as p_f:
        to_invert = pickle.load(p_f)
    pd_l = m_r["posedirs"].copy()
    pd_l[to_invert == 0] = -pd_l[to_invert == 0]
    m_l["posedirs"] = pd_l
    with open(left_path, "wb") as p_f:
        pickle.dump(m_l, p_f)
    print(f"Saved left mano file to {left_path}")

    return m_l


if __name__ == "__main__":
    RIGHT_PATH = "extra_data/mano/MANO_RIGHT.pkl"
    m_l_comp = left_from_right(RIGHT_PATH, "tmp.pkl")
    LEFT_PATH = "extra_data/mano/MANO_LEFT.pkl"
    with open(LEFT_PATH, "rb") as p_f:
        m_l = pickle.load(p_f, encoding="latin1")
    with open(RIGHT_PATH, "rb") as p_f:
        m_r = pickle.load(p_f, encoding="latin1")
    for key in m_l:
        if key == "J_regressor":
            assert np.allclose(m_l[key].toarray(), m_l_comp[key].toarray())
        elif key not in ["f", "bs_style", "shapedirs", "bs_type"]:
            assert np.allclose(m_l[key], m_l_comp[key])
