import React from 'react'
import clsx from 'clsx';
import styles from './publications.module.css'
import Link from '@docusaurus/Link';
import { publications } from '@site/static/utils/data'
import { AiFillFilePdf, AiFillGithub } from 'react-icons/ai'

const getDividedPublication = () => {
    const divPc = {}

    publications.map(publication => {
        const year = new Date(publication.date).getFullYear()
        
        if (divPc[year] === undefined) {
            divPc[year] = [publication]
        } else {
            divPc[year].push(publication)
        }
    })

    return divPc
}

const getIcon = (icon) => {
    if (icon == 'PDF')
        return <AiFillFilePdf className={styles.pc_icon}/>
    else if (icon == 'Github') 
        return <AiFillGithub className={styles.pc_icon}/>
}

const PublicationBoxContainer = ({ publication }) => {
    
    return (
        <div className={styles.pc_content}>
            <h4 className={styles.pc_title}>{publication.title}</h4>
            <Link to={publication.venue.link}>{publication.venue.name}</Link>
            {/* <p>{publication.reference}</p> */}
            <div className={styles.pc_refer}>
                {
                    Object.keys(publication.reference).map(r => 
                        <Link key={publication.reference[r]} 
                                to={publication.reference[r]} 
                                className={styles.pc_refer_link}>
                            <span>
                                {getIcon(r)}
                                {r}
                            </span>
                        </Link>
                    )
                }
            </div>
            {/* <p>{publication.tags}</p> */}
            {/* <p>{publication.type}</p> */}
            {/* <p>{publication.award}</p> */}
        </div>
    )
}

export default function PublicationListContainer() {
    const pc = getDividedPublication()
    
    return (
        <section>
            {
                Object.keys(pc).sort((a, b) => b - a).map(year => 
                    <div key={year} className={styles.pc_box}>
                        <h2 className={styles.pc_box_year}>{year}</h2>
                        <ul className={styles.pc_box_ul}>
                        {pc[year].sort((a, b) => new Date(a.date) - new Date(b.date)).map(publication => 
                            <li key={publication.title} className={styles.pc_box_li}>
                                <PublicationBoxContainer publication={publication} />
                            </li>
                        )}
                        </ul>
                    </div>
                )
            }
        </section>
    )
}
