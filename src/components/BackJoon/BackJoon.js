import React from 'react'
import clsx from 'clsx';
import styles from './backjoon.module.css'
import Link from '@docusaurus/Link';
import { backjoons } from '@site/static/utils/data'

// case 1: return "브론즈 V"
// case 2: return "브론즈 IV"
// case 3: return "브론즈 III"
// case 4: return "브론즈 IV"
// case 5: return "브론즈 I"

const getTier = (tier) => {
    let tier_text;
    let tier_style;

    switch (Math.floor(tier / 5)) {
        case 0: tier_text = '브론즈 '; break;
        case 1: tier_text = '실버 '; break;
        case 2: tier_text = '골드 '; break;
        case 3: tier_text = '플래티넘 '; break;
        case 4: tier_text = '다이아몬드 '; break;
        case 5: tier_text = '루비 '; break;
    }
    
    switch (tier % 5) {
        case 0: tier_text += 'V'; break;
        case 1: tier_text += 'IV'; break;
        case 2: tier_text += 'III'; break;
        case 3: tier_text += 'II'; break;
        case 4: tier_text += 'I'; break;
    }

    switch (Math.floor(tier / 5)) {
        case 0: tier_style = styles.tier_blonze; break;
        case 1: tier_style = styles.tier_silver; break;
        case 2: tier_style = styles.tier_gold; break;
        case 3: tier_style = styles.tier_platinum; break;
        case 4: tier_style = styles.tier_diamond; break;
        case 5: tier_style = styles.tier_ruby; break;
    }


    return <span className={`${styles.tier_span} ${tier_style}`}>{tier_text}</span>
}

const getIcon = (tier) => {
    return <img src={`https://d2gd6pc034wcta.cloudfront.net/tier/${tier}.svg`} className={styles.tier_img} />
}

const TrContainer = ({ backjoon }) => {
    return (
        <tr>
            <td className={styles.text_left}><Link to={`https://www.acmicpc.net/problem/${backjoon.id}`}>{`${backjoon.id}: ${backjoon.title}`}</Link></td>
            <td className={styles.text_left}>{getIcon(backjoon.tier)} {getTier(backjoon.tier - 1)}</td>
            <td>{backjoon.tags.map((alg, index) =>
                <span key={index}>{alg}</span>
            )}</td>
            <td>{backjoon.date}</td>
        </tr>
    )
}

export default function BackJoonTable() {
    return (
        <table>
            <thead>
                <tr>
                    <th className={styles.text_left}>Problem</th>
                    <th className={styles.text_left}>Difficulty</th>
                    <th className={styles.text_left}>Algorithm</th>
                    <th>Date</th>
                </tr>
            </thead>
            <tbody>
                {
                    backjoons.map(backjoon => 
                        <TrContainer key={backjoon.id} backjoon={backjoon} />
                    )
                }
            </tbody>
        </table>
    )
}