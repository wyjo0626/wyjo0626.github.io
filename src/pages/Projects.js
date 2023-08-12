import React from 'react'
import Layout from '@theme/Layout';
import Projects from '@site/src/components/Projects';

export default function PublicationPage() {
    return (
        <Layout title='Projects' description='Projects Page'>
            <main className={'container'}>
                <Projects />
            </main>
        </Layout>
    )
}
