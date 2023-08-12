import React from 'react'
import clsx from 'clsx';
import styles from './projects.module.css'
import Link from '@docusaurus/Link';
import { projects } from '@site/static/utils/data'
import { AiFillFilePdf, AiFillGithub } from 'react-icons/ai'

const getIcon = (icon) => {
    if (icon == 'PDF')
        return <AiFillFilePdf className={styles.pj_icon}/>
    else if (icon == 'Github') 
        return <AiFillGithub className={styles.pj_icon}/>
}

const ProjectBoxContainer = ({ project }) => {
    return (
        <div className={styles.pj_box}>
            <div className={styles.pj_img_box}>
                <picture>
                    <img sizes="(max-width: 800px) 100vw, 800px" className={styles.pj_img} src={project.image} />
                </picture>
            </div>
            <div>
                <h4 className={styles.pj_title}>{project.title}</h4>
                <span>{project.content}</span>
                <div className={styles.pc_refer}>
                    {Object.keys(project.reference).map(r => 
                        <Link key={project.reference[r]} to={project.reference[r]} className={styles.pj_refer_link}>
                            <span>
                                {getIcon(r)}
                                {r}
                            </span>
                        </Link>
                    )}
                </div>
            </div>
        </div>
    )
}

export default function ProjectListContainer() {
    return (
        <section>
            <div className={styles.pj_container}>
            {
                projects.sort((a, b) => new Date(a.date) - new Date(b.date)).map(project => 
                    <ProjectBoxContainer key={project.title} project={project} />
                )
            }
            </div>
        </section>
    )
}
