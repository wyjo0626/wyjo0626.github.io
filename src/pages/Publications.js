import React from 'react'
import Layout from '@theme/Layout';
import Publications from '@site/src/components/Publications';

export default function PublicationPage() {
    return (
        <Layout title='Publications' description='Publications Page'>
            <main className={'container'}>
                <Publications />
            </main>
        </Layout>
    )
}
